# -*- coding: utf-8 -*-
"""
Funciones puras NumPy. Este archivo NO importa QGIS ni widgets Qt.
"""
import math
import numpy as np


EPS = 1e-12


def azimuth_vector(azimuth_deg: float):
    """Azimut geológico/GIS: 0=N, 90=E. Devuelve vector XY unitario."""
    a = math.radians(float(azimuth_deg))
    return np.array([math.sin(a), math.cos(a)], dtype=float)


def pair_arrays(coords, values):
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = coords.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    dxy = coords[ju, :2] - coords[iu, :2]
    dist = np.sqrt(np.sum(dxy * dxy, axis=1))
    gamma = 0.5 * (values[ju] - values[iu]) ** 2
    good = np.isfinite(dist) & np.isfinite(gamma) & (dist > 0)
    return dxy[good], dist[good], gamma[good]


def experimental_variograms(coords, values, directions, n_lags, lag_distance, lag_tolerance, bandwidth):
    """Calcula variogramas experimentales direccionales vectorizados."""
    dxy, dist, gamma_pairs = pair_arrays(coords, values)
    results = {}
    if len(dist) == 0:
        return results

    for direction in directions:
        did = direction.get("id", "D")
        az = float(direction.get("azimuth", 0.0))
        tol_deg = float(direction.get("tolerance_angle", 22.5))
        vec = azimuth_vector(az)

        # Componente paralela al azimut, considerando simetría 180°.
        along_signed = dxy[:, 0] * vec[0] + dxy[:, 1] * vec[1]
        along = np.abs(along_signed)
        perp2 = np.maximum(dist * dist - along * along, 0.0)
        perp = np.sqrt(perp2)

        cosang = np.clip(along / np.maximum(dist, EPS), 0.0, 1.0)
        angle = np.degrees(np.arccos(cosang))
        dir_mask = (angle <= tol_deg) & (perp <= float(bandwidth))

        lag_centers = np.arange(1, int(n_lags) + 1, dtype=float) * float(lag_distance)
        gammas = np.full_like(lag_centers, np.nan, dtype=float)
        pairs = np.zeros_like(lag_centers, dtype=int)

        if np.any(dir_mask):
            h = along[dir_mask]
            g = gamma_pairs[dir_mask]
            for i, hc in enumerate(lag_centers):
                mask = np.abs(h - hc) <= float(lag_tolerance)
                pairs[i] = int(np.count_nonzero(mask))
                if pairs[i] > 0:
                    gammas[i] = float(np.mean(g[mask]))

        results[did] = {
            "azimuth": az,
            "lags": lag_centers.tolist(),
            "gamma": gammas.tolist(),
            "pairs": pairs.tolist(),
        }
    return results


def model_unit(t, model_type):
    t = np.asarray(t, dtype=float)
    mt = str(model_type).lower()
    if mt.startswith("sph"):
        return np.where(t < 1.0, 1.5 * t - 0.5 * t ** 3, 1.0)
    if mt.startswith("exp"):
        return 1.0 - np.exp(-3.0 * t)
    if mt.startswith("gau"):
        return 1.0 - np.exp(-3.0 * t * t)
    # fallback spherical
    return np.where(t < 1.0, 1.5 * t - 0.5 * t ** 3, 1.0)


def anisotropic_components(delta_xy, azimuth_deg):
    """Rota coordenadas a ejes mayor/paralelo y menor/perpendicular."""
    d = np.asarray(delta_xy, dtype=float)
    v = azimuth_vector(azimuth_deg)
    major = d[..., 0] * v[0] + d[..., 1] * v[1]
    # vector perpendicular a v
    minor = -d[..., 0] * v[1] + d[..., 1] * v[0]
    return major, minor


def variogram_gamma(delta_xy, structures, nugget, azimuth_deg=0.0, use_nugget=True):
    """Semivarianza anisotrópica multiestructura para deltas XY."""
    delta_xy = np.asarray(delta_xy, dtype=float)
    major, minor = anisotropic_components(delta_xy, azimuth_deg)
    h_euclid = np.sqrt(np.sum(delta_xy * delta_xy, axis=-1))
    gamma = np.zeros_like(h_euclid, dtype=float)

    for st in structures:
        contribution = float(st.get("contribution", 0.0))
        if contribution <= 0:
            continue
        model_type = st.get("model", "spherical")
        r_major = max(float(st.get("range_major", st.get("range_d1", 1.0))), EPS)
        r_minor = max(float(st.get("range_minor", st.get("range_d2", r_major))), EPS)
        t = np.sqrt((major / r_major) ** 2 + (minor / r_minor) ** 2)
        gamma += contribution * model_unit(t, model_type)

    if use_nugget:
        gamma += np.where(h_euclid > EPS, float(nugget), 0.0)
    return gamma


def covariance(delta_xy, structures, nugget, total_sill, azimuth_deg=0.0):
    gamma = variogram_gamma(delta_xy, structures, nugget, azimuth_deg, use_nugget=True)
    cov = float(total_sill) - gamma
    return np.maximum(cov, 0.0)


def point_in_polygon_xy(points, polygon):
    """
    Ray casting vectorizado para un anillo exterior simple.
    polygon: [[x,y], ...]. Si hay varios polígonos, usar varias llamadas y OR.
    """
    pts = np.asarray(points, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        return np.zeros(pts.shape[0], dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]
    xp = poly[:, 0]
    yp = poly[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = xp[i], yp[i]
        xj, yj = xp[j], yp[j]
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + EPS) + xi)
        inside ^= crosses
        j = i
    return inside


def points_in_any_polygon(points, polygons):
    if not polygons:
        return np.ones(np.asarray(points).shape[0], dtype=bool)
    mask = np.zeros(np.asarray(points).shape[0], dtype=bool)
    for poly in polygons:
        ring = poly.get("exterior", poly if isinstance(poly, list) else [])
        if len(ring) >= 3:
            mask |= point_in_polygon_xy(points, ring)
    return mask


def suggest_block_size(coords, values, mode="half"):
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return 1.0, 1.0, {"mean_spacing": None, "cv": None}

    # Distancia al vecino más cercano.
    # Se usa la MEDIANA, no la media, para que zonas muy densas o muy vacías
    # no deformen demasiado la sugerencia de tamaño de celda.
    nn = []
    for i in range(n):
        d = coords[:, :2] - coords[i, :2]
        dist = np.sqrt(np.sum(d * d, axis=1))
        dist[i] = np.inf
        nn.append(np.min(dist))
    nn = np.asarray(nn, dtype=float)
    typical_spacing = float(np.nanmedian(nn))
    mean_spacing = float(np.nanmean(nn))
    mean_v = float(np.mean(values)) if np.mean(values) != 0 else EPS
    cv = float(np.std(values) / abs(mean_v))

    factor = 0.5 if mode == "half" else 2.0
    # Datos muy variables: un poco más fino, pero sin exagerar.
    if cv > 1.0:
        factor *= 0.8
    elif cv < 0.35:
        factor *= 1.2
    cell = max(typical_spacing * factor, typical_spacing * 0.2, EPS)
    return float(cell), float(cell), {"typical_spacing_median_nn": typical_spacing, "mean_spacing": mean_spacing, "cv": cv}


def circular_mean_degrees(angles_deg, weights=None, default=0.0):
    angles = np.asarray(angles_deg, dtype=float)
    good = np.isfinite(angles)
    if not np.any(good):
        return float(default)
    a = np.radians(angles[good])
    if weights is None:
        weights = np.ones_like(a)
    else:
        weights = np.asarray(weights, dtype=float)[good]
    s = np.sum(np.sin(a) * weights)
    c = np.sum(np.cos(a) * weights)
    if abs(s) + abs(c) < EPS:
        return float(default)
    deg = np.degrees(np.arctan2(s, c))
    return float((deg + 360.0) % 360.0)


def axial_mean_degrees(angles_deg, weights=None, default=0.0):
    """Media circular AXIAL para orientaciones donde 0° y 180° son equivalentes.

    En anisotropía geológica un eje no tiene sentido de avance: 030° y 210°
    representan el mismo lineamiento. Por eso se promedian ángulos dobles y
    luego se dividen entre dos.
    """
    angles = np.asarray(angles_deg, dtype=float)
    good = np.isfinite(angles)
    if not np.any(good):
        return float(default)
    a2 = np.radians(2.0 * angles[good])
    if weights is None:
        w = np.ones_like(a2)
    else:
        w = np.asarray(weights, dtype=float)[good]
    s = np.sum(np.sin(a2) * w)
    c = np.sum(np.cos(a2) * w)
    if abs(s) + abs(c) < EPS:
        return float(default)
    deg = 0.5 * np.degrees(np.arctan2(s, c))
    return float((deg + 180.0) % 180.0)
