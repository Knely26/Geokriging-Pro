# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import tempfile


@dataclass
class PluginRuntimeState:
    """Estado ligero del plugin. Los datos pesados viven en archivos temporales."""
    temp_root: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="GeoKrigingPro_"))
    sample_npz: Optional[str] = None
    boundary_json: Optional[str] = None
    params_json: Optional[str] = None
    selected_variable: Optional[str] = None
    selected_layer_name: Optional[str] = None
    last_outputs: Dict[str, str] = field(default_factory=dict)
    last_payload: Dict[str, Any] = field(default_factory=dict)

    def job_dir(self, name: str) -> str:
        path = os.path.join(self.temp_root, name)
        os.makedirs(path, exist_ok=True)
        return path
