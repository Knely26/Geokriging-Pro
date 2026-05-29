# -*- coding: utf-8 -*-
import json
import os
import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, data) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_samples_npz(path: str, coords, values, lva_azimuth=None) -> str:
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    if lva_azimuth is None:
        lva_azimuth = np.full(values.shape, np.nan, dtype=float)
    else:
        lva_azimuth = np.asarray(lva_azimuth, dtype=float)
    np.savez_compressed(path, coords=coords, values=values, lva_azimuth=lva_azimuth)
    return path
