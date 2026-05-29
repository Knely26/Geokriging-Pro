# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import traceback
import shutil
import glob
import re

import numpy as np

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.PyQt.QtCore import Qt, QProcess
from qgis.PyQt.QtGui import QDoubleValidator, QIntValidator, QPixmap

from qgis.core import QgsProject, QgsWkbTypes, QgsRasterLayer, QgsVectorLayer
from qgis.gui import QgsMapLayerComboBox, QgsFieldComboBox
try:
    from qgis.gui import QgsFieldProxyModel
except Exception:
    QgsFieldProxyModel = None
from qgis.core import QgsMapLayerProxyModel

from ..core.io_utils import write_json, write_samples_npz


def _qt_align_center():
    """Compatibilidad Qt5/Qt6 para alinear widgets al centro.
    En Qt6, AlignCenter vive en Qt.AlignmentFlag.AlignCenter.
    En Qt5, vive como Qt.AlignCenter.
    """
    try:
        return Qt.AlignmentFlag.AlignCenter
    except AttributeError:
        return Qt.AlignCenter


def _qt_keep_aspect_ratio():
    try:
        return Qt.AspectRatioMode.KeepAspectRatio
    except AttributeError:
        return Qt.KeepAspectRatio


def _qt_smooth_transformation():
    try:
        return Qt.TransformationMode.SmoothTransformation
    except AttributeError:
        return Qt.SmoothTransformation


class GeoKrigingProDialog(QtWidgets.QDialog):
    """
    GUI del plugin.

    Regla de arquitectura:
    - La GUI solo lee datos QGIS, escribe archivos temporales y lanza QProcess.
    - El motor externo no importa QGIS ni toca widgets.
    """

    def __init__(self, iface, plugin_dir, state, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.plugin_dir = plugin_dir
        self.state = state
        self.current_process = None
        self.current_job = None
        self.current_output_dir = None
        self._stdout_buffer = ""
        self._stderr_buffer = ""

        self.setWindowTitle("GeoKrigingPro - motor externo seguro")
        self.resize(1220, 820)
        self._build_ui()
        self._connect_signals()
        self._set_default_values()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs)

        self.tab_vario = QtWidgets.QWidget()
        self.tab_kriging = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_vario, "Análisis Variográfico")
        self.tabs.addTab(self.tab_kriging, "Plan de Estimación (Kriging)")

        self._build_variography_tab()
        self._build_kriging_tab()

        bottom = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_status = QtWidgets.QLabel("Listo.")
        self.btn_cancel = QtWidgets.QPushButton("Cancelar proceso")
        self.btn_cancel.setEnabled(False)
        bottom.addWidget(self.lbl_status, 2)
        bottom.addWidget(self.progress, 2)
        bottom.addWidget(self.btn_cancel)
        main.addLayout(bottom)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        main.addWidget(self.log)

    def _build_variography_tab(self):
        root = QtWidgets.QHBoxLayout(self.tab_vario)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_scroll.setWidget(left)
        left_scroll.setMaximumWidth(460)
        root.addWidget(left_scroll)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        # Datos
        g_data = QtWidgets.QGroupBox("1. Datos y dominio")
        f_data = QtWidgets.QFormLayout(g_data)
        self.cmb_points = QgsMapLayerComboBox()
        self.cmb_points.setFilters(QgsMapLayerProxyModel.PointLayer)
        self.cmb_field = QgsFieldComboBox()
        if QgsFieldProxyModel is not None:
            self.cmb_field.setFilters(QgsFieldProxyModel.Numeric)
        self.cmb_field.setLayer(self.cmb_points.currentLayer())

        self.cmb_boundary = QgsMapLayerComboBox()
        try:
            self.cmb_boundary.setAllowEmptyLayer(True)
        except Exception:
            pass
        self.cmb_boundary.setFilters(QgsMapLayerProxyModel.PolygonLayer)

        # LVA ya no se pide como campo de buzamiento/azimut por punto.
        # Cuando se activa, el motor hace una primera estimación estándar y deriva
        # orientaciones locales 2D con un gradiente del mapa estimado.
        self.cmb_lva_field = QgsFieldComboBox()
        self.cmb_lva_field.setVisible(False)

        self.chk_filter_samples = QtWidgets.QCheckBox("Usar solo muestras dentro del límite geológico")
        self.chk_filter_samples.setChecked(True)
        self.btn_load = QtWidgets.QPushButton("Cargar/copiar datos a motor")

        f_data.addRow("Capa de puntos:", self.cmb_points)
        f_data.addRow("Variable numérica:", self.cmb_field)
        f_data.addRow("Límite geológico/polígono:", self.cmb_boundary)
        lva_note = QtWidgets.QLabel("LVA automático: no se pide buzamiento por punto; se infiere localmente desde una pre-estimación cuando se activa.")
        lva_note.setWordWrap(True)
        f_data.addRow("LVA:", lva_note)
        f_data.addRow("", self.chk_filter_samples)
        f_data.addRow("", self.btn_load)
        left_layout.addWidget(g_data)

        # Motor externo
        g_engine = QtWidgets.QGroupBox("2. Motor de cálculo")
        f_engine = QtWidgets.QFormLayout(g_engine)
        self.txt_python = QtWidgets.QLineEdit(self._default_python_executable())
        self.txt_python.setReadOnly(True)
        self.txt_python.setToolTip("Python real usado por el motor. Nunca debe ser qgis-bin.exe; si QGIS no lo encuentra, el proceso se bloquea antes de abrir otro QGIS.")
        f_engine.addRow("Python detectado:", self.txt_python)
        left_layout.addWidget(g_engine)

        # Direcciones
        g_dirs = QtWidgets.QGroupBox("3. Direcciones preferenciales, máximo 3")
        grid = QtWidgets.QGridLayout(g_dirs)
        grid.addWidget(QtWidgets.QLabel("Activa"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("ID"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Azimut"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("Tol. angular"), 0, 3)
        self.dir_checks = []
        self.dir_az = []
        self.dir_tol = []
        for i, default_az in enumerate([0.0, 90.0, 45.0], start=1):
            chk = QtWidgets.QCheckBox()
            chk.setChecked(i <= 2)
            lab = QtWidgets.QLabel(f"D{i}")
            az = QtWidgets.QDoubleSpinBox()
            az.setRange(0.0, 360.0)
            az.setDecimals(2)
            az.setValue(default_az)
            tol = QtWidgets.QDoubleSpinBox()
            tol.setRange(1.0, 90.0)
            tol.setDecimals(2)
            tol.setValue(22.5)
            grid.addWidget(chk, i, 0)
            grid.addWidget(lab, i, 1)
            grid.addWidget(az, i, 2)
            grid.addWidget(tol, i, 3)
            self.dir_checks.append(chk)
            self.dir_az.append(az)
            self.dir_tol.append(tol)
        left_layout.addWidget(g_dirs)

        # Experimental
        g_exp = QtWidgets.QGroupBox("4. Variograma experimental")
        f_exp = QtWidgets.QFormLayout(g_exp)
        self.sp_lags = QtWidgets.QSpinBox(); self.sp_lags.setRange(3, 80); self.sp_lags.setValue(12)
        self.sp_lag_dist = QtWidgets.QDoubleSpinBox(); self.sp_lag_dist.setRange(0.000001, 1e12); self.sp_lag_dist.setDecimals(4); self.sp_lag_dist.setValue(50.0)
        self.sp_lag_tol = QtWidgets.QDoubleSpinBox(); self.sp_lag_tol.setRange(0.000001, 1e12); self.sp_lag_tol.setDecimals(4); self.sp_lag_tol.setValue(25.0)
        self.sp_bandwidth = QtWidgets.QDoubleSpinBox(); self.sp_bandwidth.setRange(0.000001, 1e12); self.sp_bandwidth.setDecimals(4); self.sp_bandwidth.setValue(250.0)
        tol_row = QtWidgets.QHBoxLayout()
        tol_row.addWidget(self.sp_lag_tol)
        info = QtWidgets.QLabel("💡")
        info.setToolTip("La tolerancia de lag define el rango angular y de distancia permitido para que un par de muestras sea incluido en el cálculo de ese lag específico, actuando como una ventana de aceptación alrededor del vector de separación.")
        tol_row.addWidget(info)
        w_tol = QtWidgets.QWidget(); w_tol.setLayout(tol_row)
        self.btn_experimental = QtWidgets.QPushButton("Calcular variograma experimental")
        self.btn_varmap = QtWidgets.QPushButton("Generar mapa variográfico")
        f_exp.addRow("Número de lags:", self.sp_lags)
        f_exp.addRow("Separación de lag:", self.sp_lag_dist)
        f_exp.addRow("Tolerancia de lag:", w_tol)
        f_exp.addRow("Bandwidth:", self.sp_bandwidth)
        f_exp.addRow("", self.btn_experimental)
        f_exp.addRow("", self.btn_varmap)
        left_layout.addWidget(g_exp)

        # Modelo teórico
        g_model = QtWidgets.QGroupBox("5. Modelo teórico enlazado")
        v_model = QtWidgets.QVBoxLayout(g_model)
        form = QtWidgets.QFormLayout()
        self.sp_nugget = QtWidgets.QDoubleSpinBox(); self.sp_nugget.setRange(0.0, 1e12); self.sp_nugget.setDecimals(6); self.sp_nugget.setValue(0.0)
        self.sp_total_sill = QtWidgets.QDoubleSpinBox(); self.sp_total_sill.setRange(0.000001, 1e12); self.sp_total_sill.setDecimals(6); self.sp_total_sill.setValue(1.0)
        form.addRow("Nugget global:", self.sp_nugget)
        form.addRow("Sill total / varianza global:", self.sp_total_sill)
        v_model.addLayout(form)
        self.tbl_struct = QtWidgets.QTableWidget(0, 5)
        self.tbl_struct.setHorizontalHeaderLabels(["Modelo", "Sill contrib. global", "Rango mayor D1", "Ratio D2/D1", "Ratio D3/D1"])
        self.tbl_struct.horizontalHeader().setStretchLastSection(True)
        v_model.addWidget(self.tbl_struct)
        lbl_linked = QtWidgets.QLabel("Modelo enlazado: Nugget y sills son globales. D1 es el rango mayor; D2 y D3 se derivan por ratios de anisotropía.")
        lbl_linked.setWordWrap(True)
        lbl_linked.setStyleSheet("color: #555; font-size: 10px;")
        v_model.addWidget(lbl_linked)
        row_btns = QtWidgets.QHBoxLayout()
        self.btn_add_struct = QtWidgets.QPushButton("Añadir estructura")
        self.btn_del_struct = QtWidgets.QPushButton("Eliminar estructura")
        self.btn_refresh_model = QtWidgets.QPushButton("Actualizar gráfico teórico")
        self.btn_refresh_model.setToolTip("Redibuja el variograma experimental junto con las curvas teóricas. El modelo es enlazado: nugget y sills son globales; D2/D3 salen de ratios respecto a D1.")
        row_btns.addWidget(self.btn_add_struct); row_btns.addWidget(self.btn_del_struct)
        v_model.addLayout(row_btns)
        v_model.addWidget(self.btn_refresh_model)
        self.lbl_sill_check = QtWidgets.QLabel("Nugget + sills = ?")
        v_model.addWidget(self.lbl_sill_check)
        left_layout.addWidget(g_model)
        left_layout.addStretch(1)

        # Panel de imagen
        self.image_label = QtWidgets.QLabel("Aquí se mostrará el PNG generado por el motor externo.")
        self.image_label.setAlignment(_qt_align_center())
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setStyleSheet("QLabel { background: #f6f6f6; border: 1px solid #bbb; }")
        right.addWidget(self.image_label, 1)

    def _build_kriging_tab(self):
        root = QtWidgets.QHBoxLayout(self.tab_kriging)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_scroll.setWidget(left)
        left_scroll.setMaximumWidth(520)
        root.addWidget(left_scroll)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        g_var = QtWidgets.QGroupBox("Variable bloqueada desde variografía")
        f_var = QtWidgets.QFormLayout(g_var)
        self.txt_locked_var = QtWidgets.QLineEdit()
        self.txt_locked_var.setReadOnly(True)
        f_var.addRow("Variable:", self.txt_locked_var)
        left_layout.addWidget(g_var)

        g_grid = QtWidgets.QGroupBox("Grilla y bordes")
        f_grid = QtWidgets.QFormLayout(g_grid)
        self.sp_cell_x = QtWidgets.QDoubleSpinBox(); self.sp_cell_x.setRange(0.000001, 1e12); self.sp_cell_x.setDecimals(4); self.sp_cell_x.setValue(50.0)
        self.sp_cell_y = QtWidgets.QDoubleSpinBox(); self.sp_cell_y.setRange(0.000001, 1e12); self.sp_cell_y.setDecimals(4); self.sp_cell_y.setValue(50.0)
        self.btn_suggest_block = QtWidgets.QPushButton("Sugerir tamaño de bloque")
        self.chk_subblocks = QtWidgets.QCheckBox("Adaptar bordes con subbloques")
        self.chk_subblocks.setChecked(True)
        self.sp_subblocks = QtWidgets.QSpinBox(); self.sp_subblocks.setRange(1, 8); self.sp_subblocks.setValue(2)
        self.sp_block_disc = QtWidgets.QSpinBox(); self.sp_block_disc.setRange(1, 8); self.sp_block_disc.setValue(2)
        self.sp_block_disc.setToolTip("Discretización de soporte de bloque para el kriging. 1 = punto central; 2 = 2x2 puntos internos; 3 = 3x3. Mejora la estabilidad de la varianza frente al kriging puntual.")
        self.sp_max_cells = QtWidgets.QSpinBox(); self.sp_max_cells.setRange(1000, 2000000); self.sp_max_cells.setValue(150000)
        self.chk_pass_raster = QtWidgets.QCheckBox("Generar pass_used.tif diagnóstico")
        self.chk_pass_raster.setChecked(False)
        self.chk_pass_raster.setToolTip("Opcional: crea un raster donde 1/2/3 indica en qué pasada se estimó cada bloque. No es el modelo de ley.")
        self.chk_domain_fraction = QtWidgets.QCheckBox("Generar domain_fraction.tif diagnóstico")
        self.chk_domain_fraction.setChecked(False)
        self.chk_domain_fraction.setToolTip("Opcional: crea un raster con la fracción de cada celda dentro del dominio. Útil para auditoría de bordes, pero no es necesario para el modelo de ley.")
        f_grid.addRow("Celda X:", self.sp_cell_x)
        f_grid.addRow("Celda Y:", self.sp_cell_y)
        f_grid.addRow("", self.btn_suggest_block)
        f_grid.addRow("", self.chk_subblocks)
        f_grid.addRow("Subbloques/eje borde:", self.sp_subblocks)
        f_grid.addRow("Discretización bloque:", self.sp_block_disc)
        f_grid.addRow("Límite máx. de celdas:", self.sp_max_cells)
        f_grid.addRow("", self.chk_pass_raster)
        f_grid.addRow("", self.chk_domain_fraction)
        self.txt_output_dir = QtWidgets.QLineEdit(self._default_output_dir())
        self.btn_output_dir = QtWidgets.QPushButton("Elegir")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.txt_output_dir, 1)
        out_row.addWidget(self.btn_output_dir)
        out_widget = QtWidgets.QWidget(); out_widget.setLayout(out_row)
        f_grid.addRow("Carpeta GeoTIFF final:", out_widget)
        left_layout.addWidget(g_grid)

        g_samples = QtWidgets.QGroupBox("Límites de muestras")
        f_samples = QtWidgets.QFormLayout(g_samples)
        self.sp_min_samples = QtWidgets.QSpinBox(); self.sp_min_samples.setRange(1, 200); self.sp_min_samples.setValue(4)
        self.sp_max_samples = QtWidgets.QSpinBox(); self.sp_max_samples.setRange(1, 500); self.sp_max_samples.setValue(16)
        f_samples.addRow("Mínimo:", self.sp_min_samples)
        f_samples.addRow("Máximo:", self.sp_max_samples)
        left_layout.addWidget(g_samples)

        g_pass = QtWidgets.QGroupBox("Pasadas jerárquicas")
        v_pass = QtWidgets.QVBoxLayout(g_pass)
        self.tbl_pass = QtWidgets.QTableWidget(0, 4)
        self.tbl_pass.setHorizontalHeaderLabels(["Activa", "ID", "Radio X", "Radio Y"])
        self.tbl_pass.horizontalHeader().setStretchLastSection(True)
        v_pass.addWidget(self.tbl_pass)
        btns_pass = QtWidgets.QHBoxLayout()
        self.btn_add_pass = QtWidgets.QPushButton("Añadir pasada")
        self.btn_del_pass = QtWidgets.QPushButton("Eliminar pasada")
        btns_pass.addWidget(self.btn_add_pass); btns_pass.addWidget(self.btn_del_pass)
        v_pass.addLayout(btns_pass)
        left_layout.addWidget(g_pass)

        g_cap = QtWidgets.QGroupBox("Capping avanzado")
        f_cap = QtWidgets.QFormLayout(g_cap)
        self.cmb_capping = QtWidgets.QComboBox()
        self.cmb_capping.addItems(["none", "global", "local"])
        self.sp_global_cap = QtWidgets.QDoubleSpinBox(); self.sp_global_cap.setRange(0.0, 1e12); self.sp_global_cap.setDecimals(6); self.sp_global_cap.setValue(999999.0)
        self.cmb_local_threshold = QtWidgets.QComboBox(); self.cmb_local_threshold.addItems(["percentile", "mad_log"])
        self.sp_local_percentile = QtWidgets.QDoubleSpinBox(); self.sp_local_percentile.setRange(50.0, 99.99); self.sp_local_percentile.setDecimals(2); self.sp_local_percentile.setValue(98.0)
        self.sp_local_percentile.setToolTip("Percentil GLOBAL calculado con todas las muestras válidas del dominio. Luego la restricción se aplica bloque por bloque si la alta ley queda fuera del radio interno automático.")
        self.sp_mad_k = QtWidgets.QDoubleSpinBox(); self.sp_mad_k.setRange(1.0, 8.0); self.sp_mad_k.setDecimals(2); self.sp_mad_k.setValue(3.0)
        self.sp_mad_k.setToolTip("MAD log simple: convierte leyes positivas a log, calcula mediana y dispersión robusta; k=3 marca altas leyes muy alejadas de la mediana log global.")
        mad_row = QtWidgets.QHBoxLayout()
        mad_row.addWidget(self.sp_mad_k)
        mad_info = QtWidgets.QLabel("💡")
        mad_info.setToolTip("MAD log: sirve para detectar altas leyes extremas en distribuciones con cola alta. k=3 es un criterio robusto; k menor restringe más, k mayor restringe menos.")
        mad_row.addWidget(mad_info)
        mad_widget = QtWidgets.QWidget(); mad_widget.setLayout(mad_row)
        self.cmb_local_action = QtWidgets.QComboBox(); self.cmb_local_action.addItems(["keep", "local_cap", "ignore", "downweight"])
        self.sp_downweight = QtWidgets.QDoubleSpinBox(); self.sp_downweight.setRange(0.0, 1.0); self.sp_downweight.setSingleStep(0.05); self.sp_downweight.setDecimals(2); self.sp_downweight.setValue(0.25)
        note_cap = QtWidgets.QLabel("Capping local: el umbral de alta ley es GLOBAL, por percentil definido por el usuario o MAD log. Luego, para cada bloque, solo se restringe la muestra si supera ese umbral y está fuera del radio interno automático.")
        note_cap.setWordWrap(True)
        f_cap.addRow("Modo:", self.cmb_capping)
        f_cap.addRow("Top cut global:", self.sp_global_cap)
        f_cap.addRow("Umbral global:", self.cmb_local_threshold)
        f_cap.addRow("Percentil global:", self.sp_local_percentile)
        f_cap.addRow("MAD log k:", mad_widget)
        f_cap.addRow("Acción si outlier lejano:", self.cmb_local_action)
        f_cap.addRow("Factor reducción peso:", self.sp_downweight)
        f_cap.addRow("Nota:", note_cap)
        left_layout.addWidget(g_cap)

        g_neg = QtWidgets.QGroupBox("Tratamiento de negativos")
        f_neg = QtWidgets.QFormLayout(g_neg)
        self.cmb_negative = QtWidgets.QComboBox()
        self.cmb_negative.addItems(["local_p1", "local_min_positive", "allow_negative"])
        self.sp_negative_pct = QtWidgets.QDoubleSpinBox(); self.sp_negative_pct.setRange(0.0, 25.0); self.sp_negative_pct.setDecimals(2); self.sp_negative_pct.setValue(1.0)
        note_neg = QtWidgets.QLabel("Si el kriging genera un valor negativo, el default lo reemplaza por el percentil positivo local calculado con las muestras usadas para ese bloque. No usa lognormal kriging.")
        note_neg.setWordWrap(True)
        f_neg.addRow("Método:", self.cmb_negative)
        f_neg.addRow("Percentil local:", self.sp_negative_pct)
        f_neg.addRow("Nota:", note_neg)
        left_layout.addWidget(g_neg)

        g_lva = QtWidgets.QGroupBox("Anisotropía local, LVA")
        f_lva = QtWidgets.QFormLayout(g_lva)
        self.chk_lva = QtWidgets.QCheckBox("Activar LVA automático 2D, experimental")
        self.sp_default_azimuth = QtWidgets.QDoubleSpinBox(); self.sp_default_azimuth.setRange(0.0, 360.0); self.sp_default_azimuth.setDecimals(2); self.sp_default_azimuth.setValue(0.0)
        self.sp_lva_window = QtWidgets.QSpinBox(); self.sp_lva_window.setRange(1, 25); self.sp_lva_window.setValue(5)
        self.sp_lva_window.setToolTip("Ventana local en celdas: el algoritmo mira varias celdas alrededor de cada bloque para detectar la orientación dominante. Más grande = más estable/regional; más pequeño = más detalle/ruido.")
        self.sp_lva_conf = QtWidgets.QDoubleSpinBox(); self.sp_lva_conf.setRange(0.00, 0.95); self.sp_lva_conf.setDecimals(2); self.sp_lva_conf.setSingleStep(0.05); self.sp_lva_conf.setValue(0.18)
        self.sp_lva_conf.setToolTip("Confianza mínima LVA: 0 significa sin orientación clara; 1 significa lineamiento muy claro. Si una celda queda bajo este valor, usa el azimut global de respaldo.")
        self.sp_lva_smooth = QtWidgets.QSpinBox(); self.sp_lva_smooth.setRange(1, 8); self.sp_lva_smooth.setValue(2)
        self.sp_lva_smooth.setToolTip("Suavizado axial del campo angular. No promedia grados simples: trata 0° y 180° como el mismo eje de continuidad.")
        self.sp_lva_vector_step = QtWidgets.QSpinBox(); self.sp_lva_vector_step.setRange(1, 100); self.sp_lva_vector_step.setValue(5)
        self.sp_lva_vector_step.setToolTip("Cada cuántas celdas exportar una flecha/vector LVA para auditoría visual. Valores bajos generan más líneas.")
        f_lva.addRow("", self.chk_lva)
        f_lva.addRow("Azimut global de respaldo:", self.sp_default_azimuth)
        f_lva.addRow("Ventana local, celdas:", self.sp_lva_window)
        f_lva.addRow("Confianza mínima:", self.sp_lva_conf)
        f_lva.addRow("Suavizado angular:", self.sp_lva_smooth)
        f_lva.addRow("Paso vectores auditoría:", self.sp_lva_vector_step)
        note_lva = QtWidgets.QLabel("Método mejorado: pre-kriging estándar → gradientes → tensor de orientación en ventana local → confianza LVA → fallback al azimut global si la confianza es baja → suavizado axial → reorientación local del variograma. Exporta lva_azimuth, lva_confidence y lva_vectors.")
        note_lva.setWordWrap(True)
        f_lva.addRow("Nota:", note_lva)
        left_layout.addWidget(g_lva)

        self.btn_run_kriging = QtWidgets.QPushButton("Ejecutar kriging y cargar raster")
        self.btn_run_kriging.setMinimumHeight(38)
        left_layout.addWidget(self.btn_run_kriging)
        left_layout.addStretch(1)

        self.summary = QtWidgets.QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setText("Resumen de resultados de kriging.")
        right.addWidget(self.summary, 1)

    def _connect_signals(self):
        self.cmb_points.layerChanged.connect(self._on_point_layer_changed)
        self.cmb_field.fieldChanged.connect(self._on_variable_changed)
        self.btn_load.clicked.connect(self.load_data_to_temp)
        self.btn_experimental.clicked.connect(lambda: self.start_job("experimental"))
        self.btn_varmap.clicked.connect(lambda: self.start_job("varmap"))
        self.btn_suggest_block.clicked.connect(lambda: self.start_job("suggest"))
        self.btn_run_kriging.clicked.connect(lambda: self.start_job("kriging"))
        self.btn_cancel.clicked.connect(self.cancel_process)
        self.btn_output_dir.clicked.connect(self._browse_output_dir)
        self.btn_add_struct.clicked.connect(self.add_structure_row)
        self.btn_del_struct.clicked.connect(self.delete_structure_row)
        self.btn_refresh_model.clicked.connect(lambda: self.start_job("experimental"))
        self.btn_add_pass.clicked.connect(self.add_pass_row)
        self.btn_del_pass.clicked.connect(self.delete_pass_row)
        self.sp_nugget.valueChanged.connect(self.update_sill_check)
        self.sp_total_sill.valueChanged.connect(self.update_sill_check)
        self.tbl_struct.itemChanged.connect(lambda _item: self.update_sill_check())

    def _set_default_values(self):
        self.add_structure_row(model="spherical", contribution=1.0, range_major=500.0, ratio_minor=0.60, ratio_vertical=0.30)
        self.add_pass_row(active=True, pass_id=1, rx=500.0, ry=300.0)
        self.add_pass_row(active=True, pass_id=2, rx=800.0, ry=500.0)
        self.add_pass_row(active=True, pass_id=3, rx=1200.0, ry=800.0)
        self._on_variable_changed(self.cmb_field.currentField())
        self.update_sill_check()

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------
    def _default_output_dir(self):
        """Carpeta permanente para GeoTIFFs finales. No se elimina al cerrar."""
        try:
            home = QgsProject.instance().homePath()
        except Exception:
            home = ""
        if not home:
            home = os.path.join(os.path.expanduser("~"), "GeoKrigingPro_outputs")
        out = os.path.join(home, "GeoKrigingPro_outputs") if os.path.basename(home) != "GeoKrigingPro_outputs" else home
        try:
            os.makedirs(out, exist_ok=True)
        except Exception:
            out = tempfile.gettempdir()
        return out

    def _browse_output_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Carpeta para GeoTIFF final", self.txt_output_dir.text() or self._default_output_dir())
        if folder:
            self.txt_output_dir.setText(folder)

    def _looks_like_python(self, path):
        if not path or not os.path.exists(path):
            return False
        base = os.path.basename(path).lower()
        if base.startswith("qgis") or "qgis-bin" in base:
            return False
        return base.startswith("python") and (base.endswith(".exe") or base in ("python", "python3"))

    def _default_python_executable(self):
        """Detecta python real de QGIS/OSGeo4W, nunca qgis-bin.exe.

        En Windows, dentro de QGIS sys.executable puede apuntar a qgis-bin.exe.
        Si usamos eso con QProcess, se abre otra instancia de QGIS y los argumentos
        --job/--payload se interpretan como archivos de proyecto. Por eso filtramos
        qgis-bin y buscamos python.exe/python3.exe dentro de OSGeo4W/apps/Python*.
        """
        candidates = []

        def add(path):
            if path and path not in candidates:
                candidates.append(path)

        exe = getattr(sys, "executable", "") or ""
        add(exe)
        for prefix in [getattr(sys, "prefix", ""), getattr(sys, "base_prefix", "")]:
            if prefix:
                if os.name == "nt":
                    add(os.path.join(prefix, "python.exe"))
                    add(os.path.join(prefix, "python3.exe"))
                else:
                    add(os.path.join(prefix, "bin", "python3"))
                    add(os.path.join(prefix, "bin", "python"))

        search_roots = []
        if exe:
            bin_dir = os.path.dirname(exe)
            search_roots.extend([bin_dir, os.path.dirname(bin_dir)])
        for env_name in ["OSGEO4W_ROOT", "OSGEO4W64_ROOT"]:
            val = os.environ.get(env_name)
            if val:
                search_roots.append(val)
        # Caso típico del usuario: .../OSGeo4W/bin/qgis-bin.exe
        for root in list(dict.fromkeys([r for r in search_roots if r])):
            add(os.path.join(root, "python.exe"))
            add(os.path.join(root, "python3.exe"))
            add(os.path.join(root, "bin", "python.exe"))
            add(os.path.join(root, "bin", "python3.exe"))
            for pat in [os.path.join(root, "apps", "Python*", "python.exe"),
                        os.path.join(root, "apps", "Python*", "python3.exe")]:
                for hit in glob.glob(pat):
                    add(hit)

        for c in candidates:
            if self._looks_like_python(c):
                return c
        return ""

    def _on_point_layer_changed(self, layer):
        self.cmb_field.setLayer(layer)
        self.cmb_lva_field.setLayer(layer)
        self._on_variable_changed(self.cmb_field.currentField())

    def _on_variable_changed(self, field_name):
        field_name = field_name or ""
        self.state.selected_variable = field_name
        self.txt_locked_var.setText(field_name)

    def append_log(self, text):
        self.log.append(str(text))
        try:
            end_pos = QtGui.QTextCursor.MoveOperation.End
        except AttributeError:
            end_pos = QtGui.QTextCursor.End
        self.log.moveCursor(end_pos)

    def load_data_to_temp(self):
        try:
            layer = self.cmb_points.currentLayer()
            if layer is None:
                raise RuntimeError("Selecciona una capa de puntos.")
            field = self.cmb_field.currentField()
            if not field:
                raise RuntimeError("Selecciona una variable numérica.")
            coords, values, lva = self._extract_points(layer, field, None)
            if len(values) < 3:
                raise RuntimeError("Se necesitan al menos 3 puntos válidos.")
            os.makedirs(self.state.temp_root, exist_ok=True)
            sample_npz = os.path.join(self.state.temp_root, "samples.npz")
            write_samples_npz(sample_npz, coords, values, lva)
            self.state.sample_npz = sample_npz
            self.state.selected_layer_name = layer.name()
            self.state.selected_variable = field
            self.txt_locked_var.setText(field)

            boundary_json = os.path.join(self.state.temp_root, "boundary.json")
            boundary_data = self._extract_boundary_json()
            write_json(boundary_json, boundary_data)
            self.state.boundary_json = boundary_json

            data_var = float(np.nanvar(values)) if np.nanvar(values) > 0 else 1.0
            self.sp_total_sill.setValue(data_var)
            self._auto_scale_default_structure(data_var)
            self.append_log(f"Datos copiados: {len(values)} muestras válidas. Varianza global={data_var:.6g}. Temp: {self.state.temp_root}")
            self.lbl_status.setText("Datos cargados al motor externo.")
            self.update_sill_check()
        except Exception as exc:
            self._show_error("Error al cargar datos", exc)

    def _extract_points(self, layer, value_field, lva_field=None):
        coords = []
        values = []
        lva = []
        value_idx = layer.fields().indexFromName(value_field)
        if value_idx < 0:
            raise RuntimeError(f"Campo no encontrado: {value_field}")
        lva_idx = layer.fields().indexFromName(lva_field) if lva_field else -1
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                if geom.isMultipart():
                    pts = geom.asMultiPoint()
                    if not pts:
                        continue
                    pt = pts[0]
                else:
                    pt = geom.asPoint()
            except Exception:
                continue
            try:
                val = float(feat[value_idx])
            except Exception:
                continue
            if not np.isfinite(val):
                continue
            coords.append([float(pt.x()), float(pt.y())])
            values.append(val)
            if lva_idx >= 0:
                try:
                    az = float(feat[lva_idx])
                except Exception:
                    az = np.nan
            else:
                az = np.nan
            lva.append(az)
        return np.asarray(coords, dtype=float), np.asarray(values, dtype=float), np.asarray(lva, dtype=float)

    def _extract_boundary_json(self):
        layer = self.cmb_boundary.currentLayer()
        if layer is None:
            return {"polygons": []}
        polygons = []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                if geom.isMultipart():
                    mp = geom.asMultiPolygon()
                    for poly in mp:
                        if poly and poly[0]:
                            polygons.append({"exterior": [[float(p.x()), float(p.y())] for p in poly[0]]})
                else:
                    poly = geom.asPolygon()
                    if poly and poly[0]:
                        polygons.append({"exterior": [[float(p.x()), float(p.y())] for p in poly[0]]})
            except Exception:
                continue
        return {"polygons": polygons, "layer_name": layer.name()}

    # ------------------------------------------------------------------
    # Model tables
    # ------------------------------------------------------------------
    def _auto_scale_default_structure(self, data_var):
        """Ajusta la estructura inicial al orden de magnitud de la varianza.

        El variograma teórico se dibuja en las mismas unidades que la semivarianza.
        Si la estructura por defecto queda con sill=1 mientras la varianza de los
        datos es 3000, la curva teórica existe pero queda pegada al eje X y parece
        invisible. Solo autoajustamos cuando detectamos la fila inicial por defecto.
        """
        try:
            if self.tbl_struct.rowCount() != 1:
                return
            item = self.tbl_struct.item(0, 1)
            if item is None:
                return
            current = float(item.text())
            # Fila inicial creada por defecto: contribution=1.0.
            if abs(current - 1.0) <= 1e-9 and data_var > 1.0:
                nugget = float(self.sp_nugget.value())
                new_contrib = max(float(data_var) - nugget, 0.0)
                item.setText(str(new_contrib))
                self.append_log(f"Sill de la estructura inicial ajustado automáticamente a {new_contrib:.6g} para que el modelo teórico sea visible.")
        except Exception:
            pass

    def add_structure_row(self, model="spherical", contribution=0.0, range_major=100.0, ratio_minor=0.60, ratio_vertical=0.30):
        """Añade una estructura anidada del modelo teórico enlazado.

        En un variograma anisotrópico correcto no se crean sills independientes
        por dirección. Cada estructura tiene un sill global y un rango mayor.
        Las demás direcciones se derivan por ratios de anisotropía:

            range_D2 = range_D1 * ratio_minor
            range_D3 = range_D1 * ratio_vertical

        Esto imita mejor el comportamiento tipo SGeMS: al modificar la
        estructura, las curvas de todas las direcciones pertenecen al mismo
        modelo, no a modelos independientes.
        """
        row = self.tbl_struct.rowCount()
        self.tbl_struct.insertRow(row)
        combo = QtWidgets.QComboBox()
        combo.addItems(["spherical", "exponential", "gaussian"])
        combo.setCurrentText(model)
        self.tbl_struct.setCellWidget(row, 0, combo)
        for col, val in enumerate([contribution, range_major, ratio_minor, ratio_vertical], start=1):
            item = QtWidgets.QTableWidgetItem(str(float(val)))
            self.tbl_struct.setItem(row, col, item)
        combo.currentTextChanged.connect(lambda _x: self.update_sill_check())
        self.update_sill_check()

    def delete_structure_row(self):
        row = self.tbl_struct.currentRow()
        if row >= 0:
            self.tbl_struct.removeRow(row)
            self.update_sill_check()

    def get_structures(self):
        """Lee la tabla de estructuras como modelo anisotrópico enlazado.

        Columnas:
        - Sill contrib. global: aporte al sill total, compartido por todas las direcciones.
        - Rango mayor D1: rango sobre la dirección principal.
        - Ratio D2/D1 y Ratio D3/D1: relaciones de anisotropía, no sills independientes.
        """
        structures = []
        for row in range(self.tbl_struct.rowCount()):
            combo = self.tbl_struct.cellWidget(row, 0)
            model = combo.currentText() if combo else "spherical"
            try:
                contribution = float(self.tbl_struct.item(row, 1).text())
                r_major = float(self.tbl_struct.item(row, 2).text())
                ratio_minor = float(self.tbl_struct.item(row, 3).text())
                ratio_vertical = float(self.tbl_struct.item(row, 4).text())
            except Exception:
                continue

            r_major = max(r_major, 1e-12)
            # D1 se interpreta como eje mayor. Por coherencia tipo SGeMS/Isatis,
            # las razones D2/D1 y D3/D1 no deben superar 1.0. Si el usuario
            # coloca >1, internamente se limita a 1.0 para no invertir el eje mayor.
            ratio_minor = min(max(ratio_minor, 1e-12), 1.0)
            ratio_vertical = min(max(ratio_vertical, 1e-12), 1.0)
            r_minor = r_major * ratio_minor
            r_vertical = r_major * ratio_vertical

            structures.append({
                "model": model,
                "contribution": contribution,
                "range_major": r_major,
                "range_minor": r_minor,
                "range_vertical": r_vertical,
                "anis_ratio_minor": ratio_minor,
                "anis_ratio_vertical": ratio_vertical,
                "range_d1": r_major,
                "range_d2": r_minor,
                "range_d3": r_vertical,
            })
        return structures

    def update_sill_check(self):
        try:
            nugget = float(self.sp_nugget.value())
            total = float(self.sp_total_sill.value())
            sills = sum(float(s.get("contribution", 0.0)) for s in self.get_structures())
            balance = nugget + sills
            diff = balance - total
            ok = abs(diff) <= max(total, 1.0) * 0.01
            color = "#227a22" if ok else "#b00020"
            self.lbl_sill_check.setText(
                f"Nugget + ΣSills = {balance:.6g} | Sill total = {total:.6g} | Diferencia = {diff:.6g}"
            )
            self.lbl_sill_check.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception:
            self.lbl_sill_check.setText("No se pudo validar balance de sill.")

    def add_pass_row(self, active=True, pass_id=None, rx=100.0, ry=100.0):
        row = self.tbl_pass.rowCount()
        self.tbl_pass.insertRow(row)
        chk = QtWidgets.QCheckBox()
        chk.setChecked(bool(active))
        self.tbl_pass.setCellWidget(row, 0, chk)
        if pass_id is None:
            pass_id = row + 1
        for col, val in [(1, int(pass_id)), (2, float(rx)), (3, float(ry))]:
            self.tbl_pass.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))

    def delete_pass_row(self):
        row = self.tbl_pass.currentRow()
        if row >= 0:
            self.tbl_pass.removeRow(row)

    def get_passes(self):
        passes = []
        for row in range(self.tbl_pass.rowCount()):
            chk = self.tbl_pass.cellWidget(row, 0)
            try:
                pid = int(float(self.tbl_pass.item(row, 1).text()))
                rx = float(self.tbl_pass.item(row, 2).text())
                ry = float(self.tbl_pass.item(row, 3).text())
            except Exception:
                continue
            passes.append({"id": pid, "radius_x": rx, "radius_y": ry, "enabled": bool(chk.isChecked() if chk else True)})
        return passes

    def get_directions(self):
        dirs = []
        for i in range(3):
            if self.dir_checks[i].isChecked():
                dirs.append({
                    "id": f"D{i+1}",
                    "azimuth": float(self.dir_az[i].value()),
                    "tolerance_angle": float(self.dir_tol[i].value()),
                })
        if not dirs:
            raise RuntimeError("Activa al menos una dirección variográfica.")
        return dirs[:3]

    # ------------------------------------------------------------------
    # Process payload / QProcess
    # ------------------------------------------------------------------
    def build_payload(self):
        if not self.state.sample_npz or not os.path.exists(self.state.sample_npz):
            self.load_data_to_temp()
        if not self.state.sample_npz or not os.path.exists(self.state.sample_npz):
            raise RuntimeError("Primero carga/copía datos válidos al motor.")
        layer = self.cmb_points.currentLayer()
        crs_wkt = ""
        try:
            crs_wkt = layer.crs().toWkt() if layer is not None and layer.crs().isValid() else ""
        except Exception:
            crs_wkt = ""
        directions = self.get_directions()
        # El modelo anisotrópico teórico se orienta con D1 como eje mayor,
        # salvo que el usuario active LVA automático, donde este azimut se usa
        # como respaldo para celdas sin orientación local confiable.
        default_azimuth = float(directions[0].get("azimuth", self.sp_default_azimuth.value()))
        payload = {
            "sample_npz": self.state.sample_npz,
            "boundary_json": self.state.boundary_json,
            "filter_samples_by_boundary": bool(self.chk_filter_samples.isChecked()),
            "crs_wkt": crs_wkt,
            "variable": self.state.selected_variable,
            "directions": directions,
            "experimental": {
                "n_lags": int(self.sp_lags.value()),
                "lag_distance": float(self.sp_lag_dist.value()),
                "lag_tolerance": float(self.sp_lag_tol.value()),
                "bandwidth": float(self.sp_bandwidth.value()),
            },
            "varmap": {"max_pairs_plot": 80000},
            "nugget": float(self.sp_nugget.value()),
            "total_sill": float(self.sp_total_sill.value()),
            "default_azimuth": default_azimuth,
            "structures": self.get_structures(),
            "kriging": {
                "final_output_dir": self.txt_output_dir.text().strip() or self._default_output_dir(),
                "lva_mode": "auto_gradient_smoothed",
                "cell_x": float(self.sp_cell_x.value()),
                "cell_y": float(self.sp_cell_y.value()),
                "min_samples": int(self.sp_min_samples.value()),
                "max_samples": int(self.sp_max_samples.value()),
                "passes": self.get_passes(),
                "use_subblocks": bool(self.chk_subblocks.isChecked()),
                "subblocks": int(self.sp_subblocks.value()),
                "block_discretization": int(self.sp_block_disc.value()),
                "max_cells": int(self.sp_max_cells.value()),
                "use_lva": bool(self.chk_lva.isChecked()),
                "lva_smooth_iterations": int(self.sp_lva_smooth.value()),
                "lva_window_radius_cells": int(self.sp_lva_window.value()),
                "lva_confidence_threshold": float(self.sp_lva_conf.value()),
                "lva_vector_step": int(self.sp_lva_vector_step.value()),
                "output_pass_raster": bool(self.chk_pass_raster.isChecked()),
                "output_domain_fraction": bool(self.chk_domain_fraction.isChecked()),
                "negative_policy": {
                    "method": "none" if self.cmb_negative.currentText() == "allow_negative" else self.cmb_negative.currentText(),
                    "percentile": float(self.sp_negative_pct.value()),
                },
            },
            "capping": {
                "mode": self.cmb_capping.currentText(),
                "global_cap": float(self.sp_global_cap.value()),
                "local_threshold_method": self.cmb_local_threshold.currentText(),
                "global_percentile": float(self.sp_local_percentile.value()),
                "local_percentile": float(self.sp_local_percentile.value()),  # compatibilidad interna
                "mad_k": float(self.sp_mad_k.value()),
                "local_action": self.cmb_local_action.currentText(),
                "downweight_factor": float(self.sp_downweight.value()),
            },
        }
        return payload

    def start_job(self, job):
        if self.current_process is not None:
            QtWidgets.QMessageBox.warning(self, "Proceso activo", "Ya hay un proceso ejecutándose.")
            return
        try:
            payload = self.build_payload()
            out_dir = self.state.job_dir(job + "_" + QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss_zzz"))
            payload_path = os.path.join(out_dir, "payload.json")
            write_json(payload_path, payload)
            self.current_job = job
            self.current_output_dir = out_dir
            self._stdout_buffer = ""
            self._stderr_buffer = ""

            runner = os.path.join(self.plugin_dir, "engine", "runner.py")
            python_exec = self.txt_python.text().strip() or self._default_python_executable()
            if not self._looks_like_python(python_exec):
                raise RuntimeError(
                    "No se encontró un python.exe/python3.exe válido para el motor. "
                    "No se usará qgis-bin.exe porque abre otra instancia de QGIS. "
                    "Revisa que exista python3.exe o apps/Python*/python.exe dentro de tu instalación OSGeo4W/QGIS."
                )
            if not os.path.exists(runner):
                raise RuntimeError(f"No existe runner.py: {runner}")
            self.append_log(f"Iniciando job '{job}' con motor externo...")
            self.append_log(f"Python: {python_exec}")
            self.append_log(f"Output: {out_dir}")
            self.progress.setValue(0)
            self.lbl_status.setText(f"Ejecutando {job}...")
            self.btn_cancel.setEnabled(True)
            self._set_busy_buttons(False)

            proc = QProcess(self)
            proc.setWorkingDirectory(self.plugin_dir)
            env = QtCore.QProcessEnvironment.systemEnvironment()
            # Asegura que el paquete local sea importable.
            old_pp = env.value("PYTHONPATH", "")
            env.insert("PYTHONPATH", self.plugin_dir + (os.pathsep + old_pp if old_pp else ""))
            proc.setProcessEnvironment(env)
            proc.readyReadStandardOutput.connect(self._on_stdout)
            proc.readyReadStandardError.connect(self._on_stderr)
            proc.finished.connect(self._on_process_finished)
            proc.errorOccurred.connect(self._on_process_error)
            self.current_process = proc
            args = [runner, "--job", job, "--payload", payload_path, "--output", out_dir]
            proc.start(python_exec, args)
            if not proc.waitForStarted(3000):
                raise RuntimeError("No se pudo iniciar el proceso externo. Revisa la ruta de Python del motor.")
        except Exception as exc:
            self.current_process = None
            self.btn_cancel.setEnabled(False)
            self._set_busy_buttons(True)
            self._show_error("Error iniciando proceso", exc)

    def _set_busy_buttons(self, enabled):
        for b in [self.btn_load, self.btn_experimental, self.btn_varmap, self.btn_suggest_block, self.btn_run_kriging, self.btn_refresh_model]:
            b.setEnabled(enabled)

    def _on_stdout(self):
        if self.current_process is None:
            return
        text = bytes(self.current_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += text
        for line in text.splitlines():
            self.append_log(line)
            if line.startswith("PROGRESS"):
                try:
                    self.progress.setValue(int(line.split()[1]))
                except Exception:
                    pass
            elif line.startswith("GRID"):
                self.lbl_status.setText(line)

    def _on_stderr(self):
        if self.current_process is None:
            return
        text = bytes(self.current_process.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buffer += text
        for line in text.splitlines():
            self.append_log("ERR: " + line)

    def _on_process_error(self, error):
        self.append_log(f"Error QProcess: {error}")

    def _on_process_finished(self, exit_code, exit_status):
        job = self.current_job
        out_dir = self.current_output_dir
        self.current_process = None
        self.btn_cancel.setEnabled(False)
        self._set_busy_buttons(True)
        self.progress.setValue(100 if exit_code == 0 else 0)
        result_path = os.path.join(out_dir, "job_result.json") if out_dir else ""
        try:
            if not result_path or not os.path.exists(result_path):
                raise RuntimeError("El motor no generó job_result.json. Revisa el log.")
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "Error desconocido del motor."))
            data = result.get("result", {})
            self.state.last_outputs[job] = result_path
            self.lbl_status.setText(f"{job} terminado correctamente.")
            self.append_log(f"Job '{job}' terminado.")
            self._handle_job_result(job, data)
        except Exception as exc:
            self.lbl_status.setText(f"{job} falló.")
            self._show_error("Error del motor", exc)

    def _handle_job_result(self, job, data):
        if job in ("experimental", "varmap"):
            png = data.get("png")
            if png and os.path.exists(png):
                self._show_png(png)
            if job == "varmap":
                # Conserva mapa variográfico y CSV de bins fuera del temporal.
                saved = self._persist_simple_outputs(data, prefix="varmap")
                self.append_log("Mapa variográfico guardado: " + str(saved))
            if job == "experimental":
                self.append_log(f"Muestras usadas: {data.get('n_samples_used')}")
        elif job == "suggest":
            cx = float(data.get("cell_x"))
            cy = float(data.get("cell_y"))
            self.sp_cell_x.setValue(cx)
            self.sp_cell_y.setValue(cy)
            info = data.get("info", {})
            self.append_log(f"Bloque sugerido: X={cx:.4f}, Y={cy:.4f}, spacing={info.get('mean_spacing')}, CV={info.get('cv')}")
        elif job == "kriging":
            data = self._persist_kriging_outputs(data)
            self.summary.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            for key in ["estimate_raster", "variance_raster", "std_raster", "n_samples_raster", "capping_hits_raster", "domain_fraction_raster", "lva_azimuth_raster", "lva_confidence_raster", "pass_raster"]:
                path = data.get(key)
                if path and os.path.exists(path):
                    self._load_raster(path, key)
            vpath = data.get("lva_vectors_gpkg") or data.get("lva_vectors_csv")
            if vpath and os.path.exists(vpath):
                self._load_vector(vpath, "lva_vectors")
            audit_png = data.get("audit_png")
            if audit_png and os.path.exists(audit_png):
                self.append_log("Auditoría PNG: " + audit_png)
            self.append_log(f"Celdas estimadas: {data.get('n_estimated')} de {data.get('n_targets_inside')} dentro del dominio.")

    def _safe_name(self, text):
        text = str(text or "var")
        text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
        return text or "var"

    def _persist_simple_outputs(self, data, prefix="output"):
        out_dir = self.txt_output_dir.text().strip() or self._default_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        stamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss_zzz")
        var = self._safe_name(self.state.selected_variable)
        saved = {}
        for key in ["png", "csv", "json"]:
            src = data.get(key)
            if not src or not os.path.exists(src):
                continue
            ext = os.path.splitext(src)[1] or ".dat"
            dst = os.path.join(out_dir, f"GeoKrigingPro_{var}_{prefix}_{key}_{stamp}{ext}")
            shutil.copy2(src, dst)
            saved[key] = dst
        return saved

    def _persist_kriging_outputs(self, data):
        """Copia los GeoTIFF/ASC finales fuera del temporal. Los intermedios se borran al cerrar."""
        out_dir = self.txt_output_dir.text().strip() or self._default_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        stamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss_zzz")
        var = self._safe_name(self.state.selected_variable)
        label = {
            "estimate_raster": "estimate",
            "variance_raster": "variance",
            "domain_fraction_raster": "domain_fraction",
            "std_raster": "std",
            "n_samples_raster": "n_samples",
            "capping_hits_raster": "capping_hits",
            "lva_azimuth_raster": "lva_azimuth",
            "lva_confidence_raster": "lva_confidence",
            "lva_vectors_gpkg": "lva_vectors",
            "lva_vectors_csv": "lva_vectors",
            "audit_json": "audit",
            "audit_png": "audit",
            "pass_raster": "pass_used",
            "blocks_csv": "blocks",
        }
        new_data = dict(data)
        for key, short in label.items():
            src = data.get(key)
            if not src or not os.path.exists(src):
                continue
            ext = os.path.splitext(src)[1] or ".tif"
            dst = os.path.join(out_dir, f"GeoKrigingPro_{var}_{short}_{stamp}{ext}")
            shutil.copy2(src, dst)
            new_data[key] = dst
        new_data["final_output_dir"] = out_dir
        return new_data

    def closeEvent(self, event):
        """Al cerrar el plugin se borran temporales; los GeoTIFF finales ya fueron copiados."""
        try:
            if self.current_process is not None:
                self.current_process.kill()
                self.current_process = None
        except Exception:
            pass
        try:
            root = getattr(self.state, "temp_root", None)
            if root and os.path.isdir(root):
                shutil.rmtree(root, ignore_errors=True)
            # Deja el estado listo si el usuario vuelve a abrir el plugin en la misma sesión.
            self.state.temp_root = tempfile.mkdtemp(prefix="GeoKrigingPro_")
            self.state.sample_npz = None
            self.state.boundary_json = None
            self.state.last_outputs.clear()
        except Exception:
            pass
        event.accept()

    def _show_png(self, path):
        pix = QPixmap(path)
        if pix.isNull():
            self.image_label.setText(f"No se pudo cargar PNG:\n{path}")
            return
        aspect = _qt_keep_aspect_ratio()
        transform = _qt_smooth_transformation()
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), aspect, transform))
        self.image_label.setToolTip(path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Reescala si hay imagen actual.
        pix = self.image_label.pixmap()
        if pix is not None and not pix.isNull():
            # No hay acceso al original aquí; se conserva visualmente.
            pass

    def _load_vector(self, path, name):
        try:
            layer_name = f"GeoKrigingPro_{name}"
            if path.lower().endswith(".csv"):
                # CSV con WKT; QGIS lo puede abrir como capa delimitada.
                crs_authid = ""
                try:
                    lyr = self.cmb_points.currentLayer()
                    crs_authid = lyr.crs().authid() if lyr is not None and lyr.crs().isValid() else ""
                except Exception:
                    crs_authid = ""
                crs_part = f"&crs={crs_authid}" if crs_authid else ""
                uri = f"file:///{path}?delimiter=,&wktField=wkt{crs_part}"
                # Si el CRS real no se detecta, el GPKG será preferible.
                vlayer = QgsVectorLayer(uri, layer_name, "delimitedtext")
            else:
                vlayer = QgsVectorLayer(path, layer_name, "ogr")
            if vlayer.isValid():
                QgsProject.instance().addMapLayer(vlayer)
                self.append_log(f"Vector cargado: {path}")
            else:
                self.append_log(f"No se pudo cargar vector: {path}")
        except Exception as exc:
            self.append_log(f"Error cargando vector {path}: {exc}")

    def _load_raster(self, path, name):
        layer_name = f"GeoKrigingPro_{name}"
        raster = QgsRasterLayer(path, layer_name)
        if raster.isValid():
            QgsProject.instance().addMapLayer(raster)
            self.append_log(f"Raster cargado: {path}")
        else:
            self.append_log(f"No se pudo cargar raster: {path}")

    def cancel_process(self):
        if self.current_process is not None:
            self.current_process.kill()
            self.append_log("Proceso cancelado por el usuario.")
            self.current_process = None
            self.btn_cancel.setEnabled(False)
            self._set_busy_buttons(True)
            self.lbl_status.setText("Proceso cancelado.")

    def _show_error(self, title, exc):
        msg = str(exc)
        self.append_log(title + ": " + msg)
        self.append_log(traceback.format_exc())
        QtWidgets.QMessageBox.critical(self, title, msg)
