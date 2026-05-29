# -*- coding: utf-8 -*-
import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
try:
    from qgis.PyQt.QtGui import QAction
except ImportError:  # compatibilidad antigua
    from qgis.PyQt.QtWidgets import QAction

from .core.state import PluginRuntimeState
from .ui.main_dialog import GeoKrigingProDialog


class GeoKrigingProPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.toolbar = None
        self.menu_name = self.tr("&GeoKrigingPro")
        self.dialog = None
        self.state = PluginRuntimeState()

    def tr(self, message):
        return QCoreApplication.translate("GeoKrigingPro", message)

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "resources", "icon.png")
        action = QAction(QIcon(icon_path), self.tr("GeoKrigingPro"), self.iface.mainWindow())
        action.triggered.connect(self.run)
        action.setStatusTip(self.tr("Abrir GeoKrigingPro"))
        self.iface.addPluginToMenu(self.menu_name, action)
        self.toolbar = self.iface.addToolBar("GeoKrigingPro")
        self.toolbar.setObjectName("GeoKrigingPro")
        self.toolbar.addAction(action)
        self.actions.append(action)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.menu_name, action)
            self.iface.removeToolBarIcon(action)
        if self.toolbar is not None:
            del self.toolbar

    def run(self):
        if self.dialog is not None and self.dialog.isVisible():
            self.dialog.raise_()
            self.dialog.activateWindow()
            return
        self.dialog = GeoKrigingProDialog(
            iface=self.iface,
            plugin_dir=self.plugin_dir,
            state=self.state,
            parent=self.iface.mainWindow(),
        )
        self.dialog.show()
