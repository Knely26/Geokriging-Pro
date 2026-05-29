# -*- coding: utf-8 -*-
"""
Motor externo de GeoKrigingPro.

Este proceso NO importa QGIS ni PyQt. Puede fallar sin tumbar QGIS.
"""
import argparse
import json
import math
import os
import sys
import traceback

import numpy as np

# Permite importar core local aunque se ejecute como script.
PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from engine.geo_math import (  # noqa: E402
    covariance,
    experimental_variograms,
    points_in_any_polygon,
    suggest_block_size,
    circular_mean_degrees,
    axial_mean_degrees,
    anisotropic_components,
    model_unit,
)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_payload(payload_path):
    payload = read_json(payload_path)
    sample_npz = payload["sample_npz"]
    data = np.load(sample_npz)
    coords = np.asarray(data["coords"], dtype=float)
    values = np.asarray(data["values"], dtype=float)
    if "lva_azimuth" in data:
        lva_azimuth = np.asarray(data["lva_azimuth"], dtype=float)
    else:
        lva_azimuth = np.full(values.shape, np.nan)
    return payload, coords, values, lva_azimuth


def load_polygons(payload):
    path = payload.get("boundary_json")
    if not path or not os.path.exists(path):
        return []
    data = read_json(path)
    return data.get("polygons", [])


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def apply_global_capping(values, cap_value):
    values = np.asarray(values, dtype=float).copy()
    if cap_value is not None and np.isfinite(cap_value):
        values = np.minimum(values, float(cap_value))
    return values


def _range_for_direction(structure, direction_id):
    """Devuelve el rango que corresponde a D1/D2/D3 para graficar 1D.

    En la tabla del plugin las columnas son Range D1, Range D2, Range D3.
    Para compatibilidad con el kriging también existen range_major/minor/vertical.
    """
    did = str(direction_id or "D1").upper()
    if did.endswith("1"):
        return float(structure.get("range_d1", structure.get("range_major", 1.0)))
    if did.endswith("2"):
        return float(structure.get("range_d2", structure.get("range_minor", structure.get("range_major", 1.0))))
    if did.endswith("3"):
        return float(structure.get("range_d3", structure.get("range_vertical", structure.get("range_major", 1.0))))
    return float(structure.get("range_major", 1.0))


def theoretical_curve_1d(h, structures, nugget, direction_id):
    """Semivariograma teórico por dirección para graficar sobre el experimental."""
    h = np.asarray(h, dtype=float)
    gamma = np.zeros_like(h, dtype=float)
    for st in structures or []:
        contribution = float(st.get("contribution", 0.0))
        if contribution <= 0:
            continue
        r = max(_range_for_direction(st, direction_id), 1e-12)
        gamma += contribution * model_unit(h / r, st.get("model", "spherical"))
    # Convención de visualización: gamma(0)=0, pero para h>0 aparece nugget.
    gamma += np.where(h > 1e-12, float(nugget), 0.0)
    return gamma


def job_experimental(payload_path, output_dir):
    payload, coords, values, _lva = load_payload(payload_path)
    polygons = load_polygons(payload)
    if payload.get("filter_samples_by_boundary", True) and polygons:
        mask = points_in_any_polygon(coords[:, :2], polygons)
        coords = coords[mask]
        values = values[mask]

    params = payload.get("experimental", {})
    directions = payload.get("directions", [])
    result = experimental_variograms(
        coords,
        values,
        directions=directions,
        n_lags=int(params.get("n_lags", 12)),
        lag_distance=float(params.get("lag_distance", 10.0)),
        lag_tolerance=float(params.get("lag_tolerance", 5.0)),
        bandwidth=float(params.get("bandwidth", 999999.0)),
    )

    out_json = os.path.join(output_dir, "experimental_variograms.json")
    write_json(out_json, {"variograms": result, "n_samples_used": int(len(values))})

    # PNG headless. Experimental + modelo teórico enlazado.
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=130)

    structures = payload.get("structures", [])
    nugget = float(payload.get("nugget", 0.0))
    total_sill = float(payload.get("total_sill", np.nan))
    sum_sills = float(sum(float(st.get("contribution", 0.0)) for st in structures))
    balance = nugget + sum_sills

    max_lag_seen = 0.0
    plotted_any = False
    for did, item in result.items():
        lags = np.asarray(item["lags"], dtype=float)
        gam = np.asarray(item["gamma"], dtype=float)
        pairs = np.asarray(item.get("pairs", []), dtype=float)
        finite = np.isfinite(lags) & np.isfinite(gam)
        if np.any(finite):
            max_lag_seen = max(max_lag_seen, float(np.nanmax(lags[finite])))
        line, = ax.plot(lags, gam, marker="o", linewidth=1.2, label=f"{did} exp az={item['azimuth']:.1f}°")
        color = line.get_color()
        if structures:
            hmax = float(np.nanmax(lags)) if len(lags) else 1.0
            h = np.linspace(0.0, max(hmax, 1.0), 240)
            gth = theoretical_curve_1d(h, structures, nugget, did)
            ax.plot(h, gth, linestyle="--", linewidth=1.8, color=color, label=f"{did} teórico")
            plotted_any = True

    if np.isfinite(total_sill) and total_sill > 0:
        ax.axhline(total_sill, linestyle=":", linewidth=1.1, label="Sill total / varianza")
    if nugget > 0:
        ax.axhline(nugget, linestyle=":", linewidth=0.9, label="Nugget")

    title = "Variogramas experimentales + modelo teórico" if plotted_any else "Variogramas experimentales"
    ax.set_title(title)
    ax.set_xlabel("Distancia h")
    ax.set_ylabel("Semivarianza γ(h)")
    ax.grid(True, alpha=0.25)
    if result:
        ax.legend(fontsize=7, ncol=2)
    ax.text(
        0.01, 0.01,
        f"Nugget + ΣSills = {balance:.4g} | Sill total = {total_sill:.4g}",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.75"}
    )
    fig.tight_layout()
    out_png = os.path.join(output_dir, "experimental_variograms.png")
    fig.savefig(out_png)
    plt.close(fig)

    return {
        "json": out_json,
        "png": out_png,
        "n_samples_used": int(len(values)),
        "model_balance": balance,
        "total_sill": total_sill,
        "sum_structures_sill": sum_sills,
    }


def job_varmap(payload_path, output_dir):
    payload, coords, values, _lva = load_payload(payload_path)
    polygons = load_polygons(payload)
    if payload.get("filter_samples_by_boundary", True) and polygons:
        mask = points_in_any_polygon(coords[:, :2], polygons)
        coords = coords[mask]
        values = values[mask]

    n = len(values)
    if n < 2:
        raise RuntimeError("No hay suficientes muestras para mapa variografico.")
    iu, ju = np.triu_indices(n, k=1)
    dxy = coords[ju, :2] - coords[iu, :2]
    gam = 0.5 * (values[ju] - values[iu]) ** 2
    dist = np.sqrt(np.sum(dxy * dxy, axis=1))
    good = np.isfinite(gam) & np.isfinite(dist) & (dist > 0)
    dxy = dxy[good]
    gam = gam[good]
    dist = dist[good]

    max_pairs = int(payload.get("varmap", {}).get("max_pairs_plot", 80000))
    if len(gam) > max_pairs:
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(gam), size=max_pairs, replace=False)
        dxy = dxy[idx]
        gam = gam[idx]
        dist = dist[idx]

    # Simetría del mapa: +h y -h.
    dxy2 = np.vstack([dxy, -dxy])
    gam2 = np.concatenate([gam, gam])

    # Mapa variográfico binned circular: similar a una lectura por direcciones.
    # El color representa semivarianza media por celda de separación ΔX/ΔY.
    max_range = payload.get("varmap", {}).get("max_range", None)
    if max_range is None or not np.isfinite(float(max_range)) or float(max_range) <= 0:
        max_h = float(np.nanpercentile(dist, 95)) if len(dist) else 1.0
    else:
        max_h = float(max_range)
    max_h = max(max_h, 1e-9)
    bins = int(payload.get("varmap", {}).get("bins", 121))
    bins = max(41, min(bins, 301))
    edges = np.linspace(-max_h, max_h, bins + 1)
    sum_g, xedges, yedges = np.histogram2d(dxy2[:, 0], dxy2[:, 1], bins=[edges, edges], weights=gam2)
    cnt, _, _ = np.histogram2d(dxy2[:, 0], dxy2[:, 1], bins=[edges, edges])
    mean_g = np.divide(sum_g, cnt, out=np.full_like(sum_g, np.nan, dtype=float), where=cnt > 0)

    # Enmascara fuera del círculo para que no parezca un cuadrado de interpolación.
    centers_edge = (edges[:-1] + edges[1:]) / 2.0
    xx, yy = np.meshgrid(centers_edge, centers_edge, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    mean_g[rr > max_h] = np.nan

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 6.8), dpi=145)
    im = ax.imshow(
        mean_g.T,
        origin="lower",
        extent=[-max_h, max_h, -max_h, max_h],
        interpolation="nearest",
        aspect="equal",
    )
    # Círculos de distancia y rayos angulares.
    for frac in [0.25, 0.50, 0.75, 1.0]:
        circle = plt.Circle((0, 0), max_h * frac, fill=False, linewidth=0.7, alpha=0.55)
        ax.add_patch(circle)
        ax.text(max_h * frac / np.sqrt(2), max_h * frac / np.sqrt(2), f"{max_h*frac:.0f}", fontsize=7)
    for ang in range(0, 180, 15):
        rad = np.radians(ang)
        x = max_h * np.sin(rad)
        y = max_h * np.cos(rad)
        ax.plot([-x, x], [-y, y], linewidth=0.35, alpha=0.45)
        if ang % 30 == 0:
            ax.text(x * 1.03, y * 1.03, f"{ang}°", fontsize=7, ha="center", va="center")
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set_xlim(-max_h * 1.05, max_h * 1.05)
    ax.set_ylim(-max_h * 1.05, max_h * 1.05)
    ax.set_title("Mapa variográfico 2D circular\ncolor = semivarianza media γ(h)")
    ax.set_xlabel("ΔX")
    ax.set_ylabel("ΔY")
    fig.colorbar(im, ax=ax, label="γ(h)")
    fig.tight_layout()
    out_png = os.path.join(output_dir, "variogram_map.png")
    fig.savefig(out_png)
    plt.close(fig)

    # CSV de celdas binned para auditoría del mapa variográfico.
    out_csv = os.path.join(output_dir, "variogram_map_bins.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("delta_x,delta_y,semivariance_mean,pairs\n")
        for ix, x in enumerate(centers_edge):
            for iy, y in enumerate(centers_edge):
                if np.isfinite(mean_g[ix, iy]) and cnt[ix, iy] > 0:
                    f.write(f"{x:.10g},{y:.10g},{mean_g[ix,iy]:.10g},{int(cnt[ix,iy])}\n")

    return {"png": out_png, "csv": out_csv, "n_pairs_plotted": int(len(gam2)), "n_samples_used": int(n), "max_range": float(max_h), "bins": int(bins)}


def build_grid(coords, polygons, cell_x, cell_y, padding=0.0):
    if polygons:
        all_xy = []
        for poly in polygons:
            ring = poly.get("exterior", [])
            if ring:
                all_xy.extend(ring)
        arr = np.asarray(all_xy, dtype=float) if all_xy else coords[:, :2]
    else:
        arr = coords[:, :2]
    xmin, ymin = np.nanmin(arr[:, 0]) - padding, np.nanmin(arr[:, 1]) - padding
    xmax, ymax = np.nanmax(arr[:, 0]) + padding, np.nanmax(arr[:, 1]) + padding
    nx = int(math.ceil((xmax - xmin) / cell_x))
    ny = int(math.ceil((ymax - ymin) / cell_y))
    nx = max(nx, 1)
    ny = max(ny, 1)
    xs = xmin + (np.arange(nx) + 0.5) * cell_x
    ys = ymax - (np.arange(ny) + 0.5) * cell_y  # norte a sur para raster
    xx, yy = np.meshgrid(xs, ys)
    centers = np.column_stack([xx.ravel(), yy.ravel()])
    return centers, nx, ny, (xmin, ymax, cell_x, cell_y)


def make_boundary_mask(centers, polygons, subblocks, cell_x, cell_y):
    if not polygons:
        return np.ones(centers.shape[0], dtype=bool), np.ones(centers.shape[0], dtype=float)
    subblocks = max(int(subblocks), 1)
    if subblocks == 1:
        mask = points_in_any_polygon(centers, polygons)
        return mask, mask.astype(float)

    offsets = (np.arange(subblocks) + 0.5) / subblocks - 0.5
    ox, oy = np.meshgrid(offsets * cell_x, offsets * cell_y)
    offs = np.column_stack([ox.ravel(), oy.ravel()])
    inside_count = np.zeros(centers.shape[0], dtype=int)
    for off in offs:
        inside_count += points_in_any_polygon(centers + off, polygons).astype(int)
    frac = inside_count / float(len(offs))
    return frac > 0, frac


def local_lva_angle(target_xy, sample_xy, sample_azimuth, default_azimuth):
    good = np.isfinite(sample_azimuth)
    if not np.any(good):
        return float(default_azimuth)
    d = sample_xy[good] - target_xy
    dist = np.sqrt(np.sum(d * d, axis=1))
    # Usa los 12 más cercanos para orientación local.
    order = np.argsort(dist)[: min(12, len(dist))]
    weights = 1.0 / np.maximum(dist[order], 1e-6)
    return circular_mean_degrees(sample_azimuth[good][order], weights=weights, default=default_azimuth)


def _local_threshold(values, capping):
    """Calcula umbral GLOBAL para altas leyes.

    Aunque la acción se aplica bloque por bloque, el percentil NO es local:
    se calcula una sola vez con todas las muestras válidas del dominio.

    Métodos:
    - percentile: percentil global definido por el usuario, p.ej. P98/P99.
    - mad_log: umbral robusto global en espacio log:
      exp(median(logZ) + k * 1.4826 * MADlog).
    """
    vals = np.asarray(values, dtype=float)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if len(pos) == 0:
        return np.nan
    method = str(capping.get("local_threshold_method", "percentile")).lower()
    if method in ("mad_log", "mad", "log_mad"):
        logs = np.log(np.maximum(pos, 1e-12))
        med = float(np.median(logs))
        mad = float(np.median(np.abs(logs - med)))
        k = float(capping.get("mad_k", 3.0))
        return float(np.exp(med + k * 1.4826 * mad))
    pct = float(capping.get("global_percentile", capping.get("local_percentile", 98.0)))
    pct = min(max(pct, 50.0), 99.99)
    return float(np.nanpercentile(pos, pct))


def _auto_inner_radius_norm(pass_cfg):
    """Radio interno automático en unidades normalizadas del elipsoide.

    No se pide al usuario porque depende del tamaño de bloque y de la pasada.
    Regla:
      max(1.5 * diagonal_del_bloque_en_espacio_elipsoidal, 0.25)
      limitado a 0.50.

    Interpretación: una alta ley dentro de este radio se considera local al bloque;
    fuera de este radio, pero dentro del elipsoide, puede restringirse para evitar
    smearing de altas leyes.
    """
    rx = max(float(pass_cfg.get("radius_x", 1.0)), 1e-12)
    ry = max(float(pass_cfg.get("radius_y", rx)), 1e-12)
    cx = max(float(pass_cfg.get("cell_x", 0.0) or 0.0), 0.0)
    cy = max(float(pass_cfg.get("cell_y", 0.0) or 0.0), 0.0)
    if cx <= 0 or cy <= 0:
        return 0.25
    diag_norm = math.sqrt((cx / rx) ** 2 + (cy / ry) ** 2)
    return float(min(max(1.5 * diag_norm, 0.25), 0.50))


def _local_negative_floor(values, negative_policy, global_floor):
    """Valor piso local para reemplazar estimaciones negativas.

    El default es P1 local de las muestras positivas usadas para ese bloque.
    Si no hay suficientes positivos, usa el piso global positivo.
    """
    vals = np.asarray(values, dtype=float)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if len(pos) == 0:
        return float(global_floor)
    method = str(negative_policy.get("method", "local_p1")).lower()
    if method in ("none", "allow"):
        return np.nan
    if method in ("local_min_positive", "min_positive"):
        return float(np.nanmin(pos))
    pct = float(negative_policy.get("percentile", 1.0))
    pct = min(max(pct, 0.0), 25.0)
    return float(np.nanpercentile(pos, pct))


def prepare_values_for_block(raw_values, d_aniso, pass_cfg, capping):
    """Prepara valores locales para un bloque y marca outliers altos lejanos.

    Retorna:
      values_work, keep_mask, affected_mask_after_keep, meta

    - ignore elimina la muestra outlier lejana antes de resolver kriging.
    - local_cap reemplaza solo esa muestra por el umbral local del bloque.
    - downweight no cambia la ley; marca la muestra para reducir su peso luego.
    - keep no hace cambios aunque el outlier haya sido detectado.
    """
    values = np.asarray(raw_values, dtype=float).copy()
    mode = str(capping.get("mode", "none")).lower()
    keep = np.ones(values.shape, dtype=bool)
    affected = np.zeros(values.shape, dtype=bool)
    meta = {"threshold": None, "n_outliers": 0, "inner_radius_norm": None}

    if mode == "global":
        cap = capping.get("global_cap")
        if cap is not None and np.isfinite(float(cap)):
            values = np.minimum(values, float(cap))
        return values, keep, affected, meta

    if mode != "local":
        return values, keep, affected, meta

    threshold = capping.get("computed_global_threshold", None)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = np.nan
    if not np.isfinite(threshold):
        threshold = _local_threshold(values, capping)
    if not np.isfinite(threshold):
        return values, keep, affected, meta

    inner_norm = _auto_inner_radius_norm(pass_cfg)
    action = str(capping.get("local_action", "local_cap")).lower()
    outlier = values > threshold
    outside_inner = np.asarray(d_aniso, dtype=float) > inner_norm
    affected = outlier & outside_inner
    meta = {
        "threshold": float(threshold),
        "n_outliers": int(np.count_nonzero(affected)),
        "inner_radius_norm": float(inner_norm),
    }

    if not np.any(affected) or action == "keep":
        return values, keep, affected, meta

    if action == "ignore":
        keep = ~affected
        return values[keep], keep, affected[keep], meta

    if action in ("local_cap", "cap"):
        values[affected] = float(threshold)
        return values, keep, affected, meta

    # downweight: se resuelve kriging normal y luego se reduce peso explícitamente.
    return values, keep, affected, meta


def _downweight_kriging_weights(weights, affected, factor):
    """Reduce pesos de outliers y redistribuye el peso retirado.

    Regla auditable:
      w_outlier_new = w_outlier * factor
      peso retirado se redistribuye entre muestras no afectadas con pesos positivos.

    Esto mantiene aproximadamente la condición de suma de pesos = 1.
    """
    w = np.asarray(weights, dtype=float).copy()
    aff = np.asarray(affected, dtype=bool)
    if not np.any(aff):
        return w
    factor = min(max(float(factor), 0.0), 1.0)
    original_sum = float(np.sum(w))
    removed = np.sum(w[aff] * (1.0 - factor))
    w[aff] *= factor
    candidates = (~aff) & (w > 0)
    if np.any(candidates) and abs(removed) > 0:
        denom = float(np.sum(w[candidates]))
        if abs(denom) > 1e-12:
            w[candidates] += removed * (w[candidates] / denom)
    # Ajuste final para preservar suma original, evitando sesgo numérico.
    diff = original_sum - float(np.sum(w))
    non_aff = np.where(~aff)[0]
    if len(non_aff) > 0:
        w[non_aff] += diff / len(non_aff)
    else:
        w += diff / len(w)
    return w


def _block_discretization_points(target_xy, pass_cfg):
    """Puntos de discretización del bloque para kriging de bloque.

    Si block_discretization = 1, el cálculo es de punto en el centro de celda.
    Si es 2, 3, 4..., el vector de covarianza bloque-muestra se calcula como
    promedio de covarianzas entre cada muestra y los puntos internos del bloque.
    Esto hace más estable/suave la varianza de kriging y se parece más al soporte
    de bloque que muestran programas mineros.
    """
    n = int(pass_cfg.get("block_discretization", 1) or 1)
    n = max(1, min(n, 8))
    tx, ty = float(target_xy[0]), float(target_xy[1])
    if n <= 1:
        return np.array([[tx, ty]], dtype=float)
    cell_x = float(pass_cfg.get("cell_x", 0.0) or 0.0)
    cell_y = float(pass_cfg.get("cell_y", 0.0) or 0.0)
    if cell_x <= 0 or cell_y <= 0:
        return np.array([[tx, ty]], dtype=float)
    offs_x = ((np.arange(n) + 0.5) / n - 0.5) * cell_x
    offs_y = ((np.arange(n) + 0.5) / n - 0.5) * cell_y
    ox, oy = np.meshgrid(offs_x, offs_y)
    return np.column_stack([tx + ox.ravel(), ty + oy.ravel()]).astype(float)


def ordinary_kriging_one(target_xy, sample_xy, sample_values, sample_lva, structures, nugget, total_sill,
                         azimuth_deg, pass_cfg, min_samples, max_samples, capping, use_lva,
                         negative_policy=None, global_positive_floor=0.0):
    rx = max(float(pass_cfg.get("radius_x", 1.0)), 1e-9)
    ry = max(float(pass_cfg.get("radius_y", rx)), 1e-9)
    # Para vecindario usamos distancia elíptica normalizada con orientación local.
    az = azimuth_deg
    if use_lva:
        az = local_lva_angle(target_xy, sample_xy, sample_lva, azimuth_deg)

    delta = sample_xy - target_xy
    maj, minor = anisotropic_components(delta, az)
    d_aniso = np.sqrt((maj / rx) ** 2 + (minor / ry) ** 2)
    inside = d_aniso <= 1.0
    idx = np.where(inside)[0]
    if len(idx) < int(min_samples):
        return np.nan, np.nan, 0, {"negative_replaced": False, "n_capping_outliers": 0}

    order = idx[np.argsort(d_aniso[idx])]
    order = order[: int(max_samples)]
    if len(order) < int(min_samples):
        return np.nan, np.nan, int(len(order)), {"negative_replaced": False, "n_capping_outliers": 0}

    sel_xy = sample_xy[order]
    sel_values_raw = sample_values[order]
    sel_d = d_aniso[order]
    sel_values, keep, affected_after_keep, cap_meta = prepare_values_for_block(sel_values_raw, sel_d, pass_cfg, capping)
    sel_xy = sel_xy[keep]
    order = order[keep]
    if len(sel_values) < int(min_samples):
        return np.nan, np.nan, int(len(sel_values)), {"negative_replaced": False, "n_capping_outliers": int(cap_meta.get("n_outliers", 0) or 0)}

    m = len(sel_values)
    # Matriz de covarianza entre muestras usando misma orientación local.
    dij = sel_xy[None, :, :] - sel_xy[:, None, :]
    kij = covariance(dij, structures, nugget, total_sill, azimuth_deg=az)
    # Estabilidad numérica.
    kij = kij + np.eye(m) * max(float(total_sill), 1.0) * 1e-9

    K = np.ones((m + 1, m + 1), dtype=float)
    K[:m, :m] = kij
    K[m, m] = 0.0

    # Kriging de bloque simple: el vector c bloque-muestra se promedia sobre
    # puntos internos del bloque. Con discretización=1 equivale a kriging puntual.
    block_pts = _block_discretization_points(target_xy, pass_cfg)
    dt = sel_xy[:, None, :] - block_pts[None, :, :]
    k_mat = covariance(dt, structures, nugget, total_sill, azimuth_deg=az)
    k = np.mean(k_mat, axis=1)

    # Covarianza promedio bloque-bloque Cbb. Para discretización=1 es C(0).
    dbb = block_pts[None, :, :] - block_pts[:, None, :]
    cbb = float(np.mean(covariance(dbb, structures, nugget, total_sill, azimuth_deg=az)))

    rhs = np.empty(m + 1, dtype=float)
    rhs[:m] = k
    rhs[m] = 1.0

    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    w_ok = sol[:m]
    mu = sol[m]

    # Varianza formal del OK para el bloque y modelo teórico:
    # sigma² = Cbb - lambda^T c - mu, con la convención usada en el sistema.
    variance = float(max(cbb - np.dot(w_ok, k) - mu, 0.0))

    w = w_ok.copy()
    # Reducción explícita de peso para outliers altos lejanos, si se eligió esa acción.
    if str(capping.get("mode", "none")).lower() == "local" and str(capping.get("local_action", "")).lower() == "downweight":
        factor = float(capping.get("downweight_factor", 0.25))
        if len(affected_after_keep) == len(w):
            w = _downweight_kriging_weights(w, affected_after_keep, factor)

    estimate = float(np.dot(w, sel_values))

    neg_replaced = False
    if negative_policy is None:
        negative_policy = {"method": "local_p1", "percentile": 1.0}
    if np.isfinite(estimate) and estimate < 0 and str(negative_policy.get("method", "local_p1")).lower() not in ("none", "allow"):
        floor = _local_negative_floor(sel_values_raw, negative_policy, global_positive_floor)
        if np.isfinite(floor):
            estimate = float(floor)
            neg_replaced = True

    meta = {
        "negative_replaced": bool(neg_replaced),
        "n_capping_outliers": int(cap_meta.get("n_outliers", 0) or 0),
        "local_cap_threshold": cap_meta.get("threshold"),
        "inner_radius_norm": cap_meta.get("inner_radius_norm"),
        "ok_weights_sum": float(np.sum(w_ok)),
        "modified_weights_sum": float(np.sum(w)),
    }
    return estimate, variance, int(m), meta

def write_raster(path_base, array, xmin, ymax, cell_x, cell_y, nodata, crs_wkt=None):
    """Escribe GeoTIFF si GDAL está disponible; si no, ESRI ASCII Grid."""
    arr = np.asarray(array, dtype=np.float32)
    ny, nx = arr.shape
    tif_path = path_base + ".tif"
    try:
        from osgeo import gdal, osr  # noqa
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(tif_path, nx, ny, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
        ds.SetGeoTransform((xmin, cell_x, 0.0, ymax, 0.0, -cell_y))
        if crs_wkt:
            ds.SetProjection(crs_wkt)
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(float(nodata))
        out = arr.copy()
        out[~np.isfinite(out)] = nodata
        band.WriteArray(out)
        band.FlushCache()
        ds.FlushCache()
        ds = None
        return tif_path
    except Exception:
        asc_path = path_base + ".asc"
        # ASCII usa xllcorner/yllcorner; ymax es borde superior.
        yll = ymax - ny * cell_y
        out = arr.copy()
        out[~np.isfinite(out)] = nodata
        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(f"ncols {nx}\n")
            f.write(f"nrows {ny}\n")
            f.write(f"xllcorner {xmin}\n")
            f.write(f"yllcorner {yll}\n")
            f.write(f"cellsize {cell_x}\n")
            f.write(f"NODATA_value {nodata}\n")
            for row in out:
                f.write(" ".join(f"{float(v):.7g}" for v in row) + "\n")
        return asc_path



def write_block_csv(path, centers, estimates, variances, n_samples, pass_used, frac_inside,
                    capping_hits, lva_azimuth=None):
    """Exporta una tabla auditable de bloques estimados.

    Cada fila representa un centro de bloque dentro del dominio. Incluye ubicación,
    estimación, varianza, desviación estándar, número de muestras usadas y pasada.
    """
    std = np.sqrt(np.where(np.isfinite(variances) & (variances >= 0), variances, np.nan))
    with open(path, "w", encoding="utf-8") as f:
        f.write("x,y,estimate,kriging_variance,kriging_std,n_samples,pass_used,domain_fraction,capping_hits,lva_azimuth\n")
        for i in range(len(centers)):
            if not np.isfinite(estimates[i]):
                continue
            az = np.nan
            if lva_azimuth is not None:
                try:
                    az = float(lva_azimuth[i])
                except Exception:
                    az = np.nan
            f.write(
                f"{centers[i,0]:.10g},{centers[i,1]:.10g},"
                f"{float(estimates[i]):.10g},{float(variances[i]):.10g},{float(std[i]):.10g},"
                f"{int(n_samples[i]) if np.isfinite(n_samples[i]) else 0},"
                f"{int(pass_used[i]) if pass_used[i] > 0 else 0},"
                f"{float(frac_inside[i]):.6g},{int(capping_hits[i]) if np.isfinite(capping_hits[i]) else 0},"
                f"{az:.10g}\n"
            )
    return path



def _nan_fill_with_mean(arr):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=float), finite
    fill = float(np.nanmean(arr[finite]))
    return np.where(finite, arr, fill), finite


def _mean_filter_2d(arr, iterations=1):
    out = np.asarray(arr, dtype=float).copy()
    iterations = max(int(iterations), 1)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    return out


def _box_filter_2d(arr, radius_cells=3):
    """Promedio móvil cuadrado robusto, sin SciPy.

    Usa padding por borde y suma desplazamientos. Es suficientemente rápido para
    grillas de plugin y evita dependencias externas.
    """
    arr = np.asarray(arr, dtype=float)
    r = max(int(radius_cells), 1)
    padded = np.pad(arr, r, mode="edge")
    out = np.zeros_like(arr, dtype=float)
    count = 0
    h, w = arr.shape
    for dy in range(0, 2 * r + 1):
        for dx in range(0, 2 * r + 1):
            out += padded[dy:dy + h, dx:dx + w]
            count += 1
    return out / float(max(count, 1))


def _smooth_axial_azimuth_grid(az_grid, valid_mask, confidence=None, default_azimuth=0.0, iterations=2):
    """Suaviza orientaciones axiales, no direcciones vectoriales.

    Para LVA 0° y 180° representan el mismo eje de continuidad. Por eso se
    suaviza con ángulos dobles: sin(2a), cos(2a). La confianza pondera el
    suavizado: orientaciones claras arrastran más que orientaciones dudosas.
    """
    az = np.asarray(az_grid, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(az)
    if confidence is None:
        conf = valid.astype(float)
    else:
        conf = np.where(valid, np.asarray(confidence, dtype=float), 0.0)
        conf = np.clip(conf, 0.0, 1.0)
    if not np.any(valid):
        return np.full(az.shape, float(default_azimuth) % 180.0, dtype=float)

    rad2 = np.radians(2.0 * np.where(valid, az, float(default_azimuth)))
    s = np.sin(rad2) * conf
    c = np.cos(rad2) * conf
    weight = np.maximum(conf, valid.astype(float) * 1e-6)
    for _ in range(max(int(iterations), 1)):
        s_num = _mean_filter_2d(s, 1)
        c_num = _mean_filter_2d(c, 1)
        w_den = np.maximum(_mean_filter_2d(weight, 1), 1e-12)
        s = s_num / w_den
        c = c_num / w_den
        weight = np.maximum(w_den, 1e-6)
    out = (0.5 * np.degrees(np.arctan2(s, c)) + 180.0) % 180.0
    out[~valid_mask] = float(default_azimuth) % 180.0
    return out


def auto_lva_azimuth_from_grid(est_grid, cell_x, cell_y, default_azimuth=0.0,
                               smooth_iterations=2, window_radius_cells=5,
                               confidence_threshold=0.18):
    """Deriva un campo LVA 2D con ventana local y tensor estructural.

    Diferencia con la versión anterior:
      - Antes: miraba principalmente el gradiente celda a celda y luego suavizaba.
      - Ahora: calcula un STRUCTURE TENSOR en una ventana local circular/cuadrada
        alrededor de cada bloque. Esa ventana resume la orientación dominante de
        continuidad en el vecindario, por eso es más estable y menos sensible al
        ruido de una sola celda.

    Confianza/coherencia:
      confidence = (lambda1 - lambda2) / (lambda1 + lambda2)
      0.0 = no hay orientación preferente clara; 1.0 = lineamiento muy claro.
      Si confidence < threshold, se usa el azimut global de respaldo para evitar
      que el LVA invente orientaciones.

    Retorna:
      azimut axial 0-180° de continuidad mayor y raster de confianza 0-1.
    """
    arr = np.asarray(est_grid, dtype=float)
    finite = np.isfinite(arr)
    default_axis = float(default_azimuth) % 180.0
    if not np.any(finite):
        return np.full(arr.shape, default_axis, dtype=float), np.zeros(arr.shape, dtype=float)

    work, finite = _nan_fill_with_mean(arr)
    # Suavizado ligero de la pre-estimación para no perseguir ruido puntual.
    pre_smooth_iter = max(int(smooth_iterations), 1)
    work_smooth = _mean_filter_2d(work, iterations=pre_smooth_iter)

    # Gradiente en coordenadas mundo. Las filas del raster bajan de norte a sur.
    grad_row, grad_col = np.gradient(work_smooth)
    gx = grad_col / max(float(cell_x), 1e-12)
    gy = -grad_row / max(float(cell_y), 1e-12)

    # Tensor estructural local: promedia productos de gradientes en ventana local.
    # A=E[gx²], B=E[gy²], C=E[gx*gy].
    r = max(int(window_radius_cells), 1)
    A = _box_filter_2d(gx * gx, r)
    B = _box_filter_2d(gy * gy, r)
    C = _box_filter_2d(gx * gy, r)

    # Dirección de máximo cambio (gradiente dominante) en coordenadas x-y.
    theta_grad_x = 0.5 * np.arctan2(2.0 * C, A - B)  # desde eje X, CCW
    # Dirección de continuidad mayor = perpendicular al máximo cambio.
    theta_cont_x = theta_grad_x + np.pi / 2.0
    vx = np.cos(theta_cont_x)
    vy = np.sin(theta_cont_x)
    # Azimut geológico/GIS: atan2(Este, Norte). Queda axial 0-180.
    az = (np.degrees(np.arctan2(vx, vy)) + 180.0) % 180.0

    # Coherencia/confianza del lineamiento.
    coherence = np.sqrt((A - B) ** 2 + 4.0 * C * C) / np.maximum(A + B, 1e-20)
    coherence = np.clip(coherence, 0.0, 1.0)
    # Zonas casi planas o fuera de dominio: sin confianza.
    grad_energy = A + B
    if np.any(grad_energy[finite] > 0):
        energy_floor = float(np.nanpercentile(grad_energy[finite], 10))
        coherence[grad_energy <= energy_floor] = 0.0
    coherence[~finite] = 0.0

    # Fallback al azimut global donde la orientación es dudosa.
    conf_threshold = min(max(float(confidence_threshold), 0.0), 0.95)
    raw_az = az.copy()
    raw_az[(coherence < conf_threshold) | (~finite)] = default_axis

    # Suavizado axial ponderado por confianza.
    smooth_az = _smooth_axial_azimuth_grid(raw_az, finite, confidence=np.maximum(coherence, 1e-6),
                                           default_azimuth=default_axis,
                                           iterations=max(int(smooth_iterations), 1))
    smooth_az[(coherence < conf_threshold) | (~finite)] = default_axis
    return smooth_az, coherence


def write_lva_vectors(output_dir, centers, nx, ny, geo, azimuths, confidence, domain_mask,
                      crs_wkt=None, step=5, length_factor=3.0):
    """Exporta flechas/lineamientos LVA para auditoría visual.

    Crea CSV con WKT y, si GDAL/OGR está disponible, también GeoPackage.
    """
    xmin, ymax, cell_x, cell_y = geo
    step = max(int(step), 1)
    length = max(float(cell_x), float(cell_y)) * float(length_factor)
    csv_path = os.path.join(output_dir, "lva_vectors.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("wkt,x,y,azimuth,confidence\n")
        for row in range(0, ny, step):
            for col in range(0, nx, step):
                idx = row * nx + col
                if idx >= len(centers) or not domain_mask[idx]:
                    continue
                az = float(azimuths[idx]) if np.isfinite(azimuths[idx]) else np.nan
                conf = float(confidence[idx]) if confidence is not None and np.isfinite(confidence[idx]) else np.nan
                if not np.isfinite(az):
                    continue
                rad = math.radians(az)
                dx = math.sin(rad) * length / 2.0
                dy = math.cos(rad) * length / 2.0
                x, y = float(centers[idx, 0]), float(centers[idx, 1])
                wkt = f"LINESTRING ({x-dx:.10g} {y-dy:.10g}, {x+dx:.10g} {y+dy:.10g})"
                f.write(f'"{wkt}",{x:.10g},{y:.10g},{az:.6g},{conf:.6g}\n')

    gpkg_path = os.path.join(output_dir, "lva_vectors.gpkg")
    try:
        from osgeo import ogr, osr  # type: ignore
        if os.path.exists(gpkg_path):
            os.remove(gpkg_path)
        drv = ogr.GetDriverByName("GPKG")
        ds = drv.CreateDataSource(gpkg_path)
        srs = None
        if crs_wkt:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs_wkt)
        layer = ds.CreateLayer("lva_vectors", srs, ogr.wkbLineString)
        layer.CreateField(ogr.FieldDefn("azimuth", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn("confidence", ogr.OFTReal))
        defn = layer.GetLayerDefn()
        for row in range(0, ny, step):
            for col in range(0, nx, step):
                idx = row * nx + col
                if idx >= len(centers) or not domain_mask[idx]:
                    continue
                az = float(azimuths[idx]) if np.isfinite(azimuths[idx]) else np.nan
                conf = float(confidence[idx]) if confidence is not None and np.isfinite(confidence[idx]) else np.nan
                if not np.isfinite(az):
                    continue
                rad = math.radians(az)
                dx = math.sin(rad) * length / 2.0
                dy = math.cos(rad) * length / 2.0
                x, y = float(centers[idx, 0]), float(centers[idx, 1])
                geom = ogr.Geometry(ogr.wkbLineString)
                geom.AddPoint(x - dx, y - dy)
                geom.AddPoint(x + dx, y + dy)
                feat = ogr.Feature(defn)
                feat.SetGeometry(geom)
                feat.SetField("azimuth", az)
                feat.SetField("confidence", conf)
                layer.CreateFeature(feat)
                feat = None
        ds = None
        return gpkg_path, csv_path
    except Exception:
        return None, csv_path


def write_audit_outputs(output_dir, summary, est_grid, var_grid, std_grid, n_samples_grid,
                        pass_grid, capping_hits_grid, lva_conf_grid=None):
    """Genera JSON y PNG de auditoría rápida del kriging."""
    audit = dict(summary)
    finite_est = np.isfinite(est_grid)
    finite_var = np.isfinite(var_grid)
    audit["estimate_stats"] = {
        "min": float(np.nanmin(est_grid[finite_est])) if np.any(finite_est) else None,
        "mean": float(np.nanmean(est_grid[finite_est])) if np.any(finite_est) else None,
        "max": float(np.nanmax(est_grid[finite_est])) if np.any(finite_est) else None,
    }
    audit["variance_stats"] = {
        "min": float(np.nanmin(var_grid[finite_var])) if np.any(finite_var) else None,
        "mean": float(np.nanmean(var_grid[finite_var])) if np.any(finite_var) else None,
        "max": float(np.nanmax(var_grid[finite_var])) if np.any(finite_var) else None,
    }
    if lva_conf_grid is not None and np.any(np.isfinite(lva_conf_grid)):
        c = lva_conf_grid[np.isfinite(lva_conf_grid)]
        audit["lva_confidence_stats"] = {"min": float(np.nanmin(c)), "mean": float(np.nanmean(c)), "max": float(np.nanmax(c))}
    audit_json = os.path.join(output_dir, "kriging_audit.json")
    write_json(audit_json, audit)

    try:
        plt = setup_matplotlib()
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), dpi=130)
        ax = axes.ravel()
        vals = est_grid[np.isfinite(est_grid)]
        if len(vals):
            ax[0].hist(vals, bins=35)
        ax[0].set_title("Histograma de estimación")
        ax[0].set_xlabel("Estimate")
        svals = std_grid[np.isfinite(std_grid)]
        if len(svals):
            ax[1].hist(svals, bins=35)
        ax[1].set_title("Histograma de desviación OK")
        ax[1].set_xlabel("Kriging std")
        ns = n_samples_grid[np.isfinite(n_samples_grid)]
        if len(ns):
            ax[2].hist(ns, bins=np.arange(np.nanmin(ns), np.nanmax(ns) + 2) - 0.5)
        ax[2].set_title("Muestras usadas por bloque")
        ax[2].set_xlabel("n_samples")
        if lva_conf_grid is not None and np.any(np.isfinite(lva_conf_grid)):
            cf = lva_conf_grid[np.isfinite(lva_conf_grid)]
            ax[3].hist(cf, bins=30)
            ax[3].set_title("Confianza LVA")
            ax[3].set_xlabel("0=dudoso, 1=claro")
        else:
            ph = pass_grid[np.isfinite(pass_grid)]
            if len(ph):
                ax[3].hist(ph, bins=np.arange(np.nanmin(ph), np.nanmax(ph) + 2) - 0.5)
            ax[3].set_title("Pasada usada")
            ax[3].set_xlabel("pass_used")
        for a in ax:
            a.grid(True, alpha=0.25)
        fig.suptitle("GeoKrigingPro - auditoría rápida")
        fig.tight_layout()
        audit_png = os.path.join(output_dir, "kriging_audit.png")
        fig.savefig(audit_png)
        plt.close(fig)
    except Exception:
        audit_png = None
    return audit_json, audit_png


def job_kriging(payload_path, output_dir):
    payload, coords, values, lva = load_payload(payload_path)
    polygons = load_polygons(payload)
    sample_mask = np.isfinite(values)
    if payload.get("filter_samples_by_boundary", True) and polygons:
        sample_mask &= points_in_any_polygon(coords[:, :2], polygons)
    coords = coords[sample_mask]
    values = values[sample_mask]
    lva = lva[sample_mask]

    if len(values) < 3:
        raise RuntimeError("No hay suficientes muestras dentro del dominio para kriging.")

    capping = dict(payload.get("capping", {"mode": "none"}))
    if capping.get("mode") == "global" and capping.get("global_cap") is not None:
        values = apply_global_capping(values, float(capping.get("global_cap")))
    if str(capping.get("mode", "none")).lower() == "local":
        capping["computed_global_threshold"] = _local_threshold(values, capping)

    kr = payload.get("kriging", {})
    cell_x = float(kr.get("cell_x", 0.0) or 0.0)
    cell_y = float(kr.get("cell_y", 0.0) or 0.0)
    if cell_x <= 0 or cell_y <= 0:
        cell_x, cell_y, _info = suggest_block_size(coords, values, mode="half")

    centers, nx, ny, geo = build_grid(coords, polygons, cell_x, cell_y, padding=0.0)
    xmin, ymax, cell_x, cell_y = geo
    max_cells = int(kr.get("max_cells", 150000))
    if nx * ny > max_cells:
        raise RuntimeError(
            f"La grilla tiene {nx*ny:,} celdas ({nx} x {ny}), supera el limite {max_cells:,}. "
            f"Aumenta el tamaño de bloque."
        )

    subblocks = int(kr.get("subblocks", 1)) if kr.get("use_subblocks", True) else 1
    domain_mask, frac_inside = make_boundary_mask(centers, polygons, subblocks, cell_x, cell_y)

    structures = payload.get("structures", [])
    nugget = float(payload.get("nugget", 0.0))
    data_variance = float(np.nanvar(values)) if np.nanvar(values) > 0 else 1.0
    requested_total_sill = float(payload.get("total_sill", data_variance))
    sum_struct_sill = float(sum(float(st.get("contribution", 0.0)) for st in structures or []))
    model_total_sill = float(nugget + sum_struct_sill)
    total_sill = model_total_sill if model_total_sill > 0 else requested_total_sill
    if not structures:
        # Modelo fallback: esférico con rango aproximado.
        cell, _, _info = suggest_block_size(coords, values, mode="half")
        total_sill = max(requested_total_sill, data_variance, 1e-9)
        structures = [{"model": "spherical", "contribution": max(total_sill - nugget, 1e-9), "range_major": cell * 10, "range_minor": cell * 6}]
        sum_struct_sill = float(sum(float(st.get("contribution", 0.0)) for st in structures))
        model_total_sill = float(nugget + sum_struct_sill)
        total_sill = model_total_sill

    default_azimuth = float(payload.get("default_azimuth", 0.0))
    use_lva = bool(kr.get("use_lva", False))
    min_samples = int(kr.get("min_samples", 4))
    max_samples = int(kr.get("max_samples", 16))
    passes = kr.get("passes", [])
    if not passes:
        passes = [{"id": 1, "radius_x": cell_x * 10, "radius_y": cell_y * 10, "enabled": True}]

    # Se añade el tamaño de bloque a cada pasada para calcular radio interno automático
    # y la discretización de soporte de bloque para el vector bloque-muestra.
    block_disc = int(kr.get("block_discretization", 2) or 2)
    passes = [dict(p, cell_x=float(cell_x), cell_y=float(cell_y), block_discretization=block_disc) for p in passes]

    samples_xy = coords[:, :2]
    valid_targets = np.where(domain_mask)[0]
    total_targets = len(valid_targets)
    enabled_passes = [x for x in passes if x.get("enabled", True)]
    print(f"GRID {nx} {ny} CELLS {nx*ny} TARGETS {total_targets}", flush=True)

    pos_global = values[np.isfinite(values) & (values > 0)]
    if len(pos_global) > 0:
        global_positive_floor = float(np.nanpercentile(pos_global, 1.0))
    else:
        global_positive_floor = 0.0
    negative_policy = kr.get("negative_policy", {"method": "local_p1", "percentile": 1.0})

    def estimate_all(target_azimuths=None, emit_progress=True, stage_label="OK"):
        estimates_local = np.full(centers.shape[0], np.nan, dtype=float)
        variances_local = np.full(centers.shape[0], np.nan, dtype=float)
        n_samples_local = np.full(centers.shape[0], np.nan, dtype=float)
        capping_hits_local = np.zeros(centers.shape[0], dtype=float)
        pass_used_local = np.zeros(centers.shape[0], dtype=np.int16)
        stats = {"negative_replaced": 0, "capping_outlier_hits": 0}
        done_local = 0
        next_progress_local = 0
        denom = max(total_targets * max(1, len(enabled_passes)), 1)
        for pidx, p in enumerate(passes, start=1):
            if not p.get("enabled", True):
                continue
            remaining = valid_targets[~np.isfinite(estimates_local[valid_targets])]
            if len(remaining) == 0:
                break
            print(f"{stage_label}_PASS {pidx} remaining={len(remaining)} rx={p.get('radius_x')} ry={p.get('radius_y')}", flush=True)
            for idx in remaining:
                local_az = float(default_azimuth)
                if target_azimuths is not None:
                    try:
                        if np.isfinite(target_azimuths[idx]):
                            local_az = float(target_azimuths[idx])
                    except Exception:
                        local_az = float(default_azimuth)
                z, var, n_used, meta = ordinary_kriging_one(
                    centers[idx], samples_xy, values, lva,
                    structures, nugget, total_sill,
                    azimuth_deg=local_az,
                    pass_cfg=p,
                    min_samples=min_samples,
                    max_samples=max_samples,
                    capping=capping,
                    use_lva=False,
                    negative_policy=negative_policy,
                    global_positive_floor=global_positive_floor,
                )
                if np.isfinite(z):
                    estimates_local[idx] = z
                    variances_local[idx] = var
                    n_samples_local[idx] = float(n_used)
                    pass_used_local[idx] = int(p.get("id", pidx))
                    capping_hits_local[idx] = float(meta.get("n_capping_outliers", 0) or 0)
                    if meta.get("negative_replaced"):
                        stats["negative_replaced"] += 1
                    stats["capping_outlier_hits"] += int(meta.get("n_capping_outliers", 0) or 0)
                done_local += 1
                if emit_progress:
                    progress = int(100 * done_local / denom)
                    if progress >= next_progress_local:
                        print(f"PROGRESS {min(progress, 99)}", flush=True)
                        next_progress_local += 5
        return estimates_local, variances_local, n_samples_local, capping_hits_local, pass_used_local, stats

    az_grid = None
    conf_grid = None
    if use_lva:
        print("LVA_AUTO pre-kriging estándar para inferir orientación local 2D", flush=True)
        pre_estimates, _pre_vars, _pre_nsamp, _pre_chits, _pre_pass, _pre_stats = estimate_all(target_azimuths=None, emit_progress=False, stage_label="LVA_PRE")
        smooth_iter = int(kr.get("lva_smooth_iterations", 2))
        window_radius = int(kr.get("lva_window_radius_cells", 5))
        conf_threshold = float(kr.get("lva_confidence_threshold", 0.18))
        az_grid, conf_grid = auto_lva_azimuth_from_grid(
            pre_estimates.reshape(ny, nx), cell_x, cell_y,
            default_azimuth=default_azimuth,
            smooth_iterations=smooth_iter,
            window_radius_cells=window_radius,
            confidence_threshold=conf_threshold,
        )
        target_azimuths = az_grid.ravel()
        print(
            f"LVA_AUTO ventana={window_radius} celdas confianza_min={conf_threshold:.3f}; "
            "re-ejecutando kriging con variograma reorientado localmente",
            flush=True,
        )
        estimates, variances, n_samples_grid_flat, capping_hits_flat, pass_used, stats = estimate_all(target_azimuths=target_azimuths, emit_progress=True, stage_label="LVA")
    else:
        estimates, variances, n_samples_grid_flat, capping_hits_flat, pass_used, stats = estimate_all(target_azimuths=None, emit_progress=True, stage_label="OK")

    nodata = -9999.0
    est_grid = estimates.reshape(ny, nx)
    var_grid = variances.reshape(ny, nx)
    pass_grid = pass_used.reshape(ny, nx).astype(float)
    pass_grid[pass_grid == 0] = np.nan
    frac_grid = frac_inside.reshape(ny, nx).astype(float)

    crs_wkt = payload.get("crs_wkt")
    est_path = write_raster(os.path.join(output_dir, "kriging_estimate"), est_grid, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)
    var_path = write_raster(os.path.join(output_dir, "kriging_variance"), var_grid, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)
    std_grid = np.sqrt(np.where(np.isfinite(var_grid) & (var_grid >= 0), var_grid, np.nan))
    std_path = write_raster(os.path.join(output_dir, "kriging_std"), std_grid, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)
    n_samples_grid = n_samples_grid_flat.reshape(ny, nx).astype(float)
    capping_hits_grid = capping_hits_flat.reshape(ny, nx).astype(float)
    n_samples_path = write_raster(os.path.join(output_dir, "kriging_n_samples"), n_samples_grid, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)
    capping_hits_path = write_raster(os.path.join(output_dir, "kriging_capping_hits"), capping_hits_grid, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)
    domain_frac_path = None
    if bool(kr.get("output_domain_fraction", False)):
        domain_frac_path = write_raster(os.path.join(output_dir, "domain_fraction"), frac_grid, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)

    pass_path = None
    if bool(kr.get("output_pass_raster", False)):
        pass_path = write_raster(os.path.join(output_dir, "kriging_pass_used"), pass_grid, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)

    lva_path = None
    lva_conf_path = None
    lva_vectors_gpkg = None
    lva_vectors_csv = None
    target_azimuths_for_csv = None
    if use_lva and az_grid is not None:
        # Fuera del dominio queda NoData para revisar solo zonas estimables.
        lva_out = az_grid.astype(float)
        lva_out[~domain_mask.reshape(ny, nx)] = np.nan
        lva_path = write_raster(os.path.join(output_dir, "lva_azimuth"), lva_out, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)
        if conf_grid is not None:
            conf_out = conf_grid.astype(float)
            conf_out[~domain_mask.reshape(ny, nx)] = np.nan
            lva_conf_path = write_raster(os.path.join(output_dir, "lva_confidence"), conf_out, xmin, ymax, cell_x, cell_y, nodata, crs_wkt)
        target_azimuths_for_csv = az_grid.ravel()
        try:
            vector_step = int(kr.get("lva_vector_step", max(3, min(nx, ny) // 40 if min(nx, ny) > 0 else 5)))
            lva_vectors_gpkg, lva_vectors_csv = write_lva_vectors(
                output_dir, centers, nx, ny, geo,
                az_grid.ravel(), conf_grid.ravel() if conf_grid is not None else None,
                domain_mask, crs_wkt=crs_wkt, step=vector_step,
                length_factor=float(kr.get("lva_vector_length_factor", 3.0)),
            )
        except Exception:
            lva_vectors_gpkg, lva_vectors_csv = None, None

    block_csv_path = write_block_csv(
        os.path.join(output_dir, "kriging_blocks.csv"),
        centers, estimates, variances, n_samples_grid_flat, pass_used, frac_inside,
        capping_hits_flat, lva_azimuth=target_azimuths_for_csv
    )

    summary = {
        "estimate_raster": est_path,
        "variance_raster": var_path,
        "std_raster": std_path,
        "n_samples_raster": n_samples_path,
        "capping_hits_raster": capping_hits_path,
        "blocks_csv": block_csv_path,
        "pass_raster": pass_path,
        "domain_fraction_raster": domain_frac_path,
        "lva_azimuth_raster": lva_path,
        "lva_confidence_raster": lva_conf_path,
        "lva_vectors_gpkg": lva_vectors_gpkg,
        "lva_vectors_csv": lva_vectors_csv,
        "nx": int(nx),
        "ny": int(ny),
        "n_cells": int(nx * ny),
        "n_targets_inside": int(total_targets),
        "n_estimated": int(np.count_nonzero(np.isfinite(estimates))),
        "n_samples_used": int(len(values)),
        "cell_x": float(cell_x),
        "cell_y": float(cell_y),
        "lva_used": bool(use_lva),
        "lva_mode": "auto_window_tensor_confidence" if use_lva else "off",
        "lva_window_radius_cells": int(kr.get("lva_window_radius_cells", 5)),
        "lva_confidence_threshold": float(kr.get("lva_confidence_threshold", 0.18)),
        "lva_smooth_iterations": int(kr.get("lva_smooth_iterations", 2)),
        "negative_replaced_count": int(stats.get("negative_replaced", 0)),
        "capping_outlier_hits": int(stats.get("capping_outlier_hits", 0)),
        "data_variance": float(data_variance),
        "requested_total_sill": float(requested_total_sill),
        "effective_model_sill": float(total_sill),
        "sum_structures_sill": float(sum_struct_sill),
        "block_discretization": int(kr.get("block_discretization", 1) or 1),
        "variance_note": "OK variance from theoretical model. If LVA or downweight is active, read as diagnostic relative variance.",
        "negative_policy": negative_policy,
        "capping": capping,
    }
    # Auditoría rápida: JSON + PNG con histogramas y métricas principales.
    try:
        audit_json, audit_png = write_audit_outputs(
            output_dir, summary, est_grid, var_grid, std_grid, n_samples_grid,
            pass_grid, capping_hits_grid,
            lva_conf_grid=(conf_grid if use_lva and conf_grid is not None else None),
        )
        summary["audit_json"] = audit_json
        summary["audit_png"] = audit_png
    except Exception:
        summary["audit_json"] = None
        summary["audit_png"] = None

    write_json(os.path.join(output_dir, "kriging_summary.json"), summary)
    return summary


def job_suggest(payload_path, output_dir):
    payload, coords, values, _lva = load_payload(payload_path)
    polygons = load_polygons(payload)
    if payload.get("filter_samples_by_boundary", True) and polygons:
        mask = points_in_any_polygon(coords[:, :2], polygons)
        coords = coords[mask]
        values = values[mask]
    x, y, info = suggest_block_size(coords, values, mode=payload.get("suggest_mode", "half"))
    out = {"cell_x": x, "cell_y": y, "info": info, "n_samples_used": int(len(values))}
    write_json(os.path.join(output_dir, "suggest_block.json"), out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, choices=["experimental", "varmap", "kriging", "suggest"])
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    try:
        if args.job == "experimental":
            result = job_experimental(args.payload, args.output)
        elif args.job == "varmap":
            result = job_varmap(args.payload, args.output)
        elif args.job == "suggest":
            result = job_suggest(args.payload, args.output)
        else:
            result = job_kriging(args.payload, args.output)
        final_path = os.path.join(args.output, "job_result.json")
        write_json(final_path, {"ok": True, "result": result})
        print("JOB_OK " + final_path, flush=True)
        return 0
    except Exception as exc:
        err = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        final_path = os.path.join(args.output, "job_result.json")
        write_json(final_path, err)
        print("JOB_ERROR " + str(exc), file=sys.stderr, flush=True)
        print(err["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
