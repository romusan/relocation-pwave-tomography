#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the current SGC relocation-tomography workflow from a 1-D Vp start.

This wrapper imports the production joint relocation/tomography script and
replaces only the starting P-wave model.  The replacement model is laterally
uniform at each depth level and is computed as the domain median of the
production starting model, so it preserves the depth-dependent reference
structure but removes the smooth slab-guided lateral component.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
BASE_SCRIPT_CANDIDATES = [
    HERE.parents[2] / "codigos" / "run_sgc_new_events_joint_vp_relocation.py",
    HERE.with_name("run_sgc_new_events_joint_vp_relocation.py"),
    HERE.parents[1] / "code" / "run_sgc_new_events_joint_vp_relocation.py",
]
BASE_SCRIPT = next((path for path in BASE_SCRIPT_CANDIDATES if path.exists()), BASE_SCRIPT_CANDIDATES[0])


def import_base():
    spec = importlib.util.spec_from_file_location("joint_relocation_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base workflow: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_relocation_base"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = import_base()
    original_loader = base.load_q_grid_and_vp

    def load_depth_only_grid_and_vp():
        xg, yg, zg, vp_pref = original_loader()
        domain = base.domain_mask(xg, yg, zg)
        vp_1d = np.empty_like(vp_pref, dtype=float)
        profile = []
        for iz, z in enumerate(zg):
            vals = vp_pref[iz][domain[iz] & np.isfinite(vp_pref[iz])]
            if vals.size == 0:
                vals = vp_pref[iz][np.isfinite(vp_pref[iz])]
            depth_vp = float(np.nanmedian(vals))
            vp_1d[iz, :, :] = depth_vp
            profile.append({"depth_km": float(z), "vp_1d_km_s": depth_vp})
        base.DATA.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            base.DATA / "current_dataset_1d_starting_profile.csv",
            np.array([[p["depth_km"], p["vp_1d_km_s"]] for p in profile], dtype=float),
            delimiter=",",
            header="depth_km,vp_1d_km_s",
            comments="",
        )
        return xg, yg, zg, vp_1d

    base.load_q_grid_and_vp = load_depth_only_grid_and_vp
    base.main()


if __name__ == "__main__":
    os.environ.setdefault("MIN_PICKS_PER_EVENT", "4")
    os.environ.setdefault("TRAVEL_SOLVER", "fmm")
    main()
