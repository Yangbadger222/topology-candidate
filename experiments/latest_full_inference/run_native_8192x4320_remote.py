#!/usr/bin/env python3
"""Run the frozen complete MaGRoad graph pipeline at 8192 x 4320."""

from __future__ import annotations

import sys

import numpy as np

import run_latest_full_inference as inference


if __name__ == "__main__":
    # The checkpoint was serialized by NumPy 2.x but the validated inference
    # environment uses NumPy 1.24. Restore pickle module names after imports.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    inference.INFERENCE_SIZE = (8192, 4320)
    inference.main()
