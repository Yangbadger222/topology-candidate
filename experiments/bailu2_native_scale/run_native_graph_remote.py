#!/usr/bin/env python3
"""Run the frozen complete MaGRoad graph pipeline at bailu_2 native size."""

from __future__ import annotations

import run_latest_full_inference as inference


if __name__ == "__main__":
    import sys

    import numpy as np

    # The checkpoint was serialized by NumPy 2.x but the validated inference
    # environment uses NumPy 1.24. Restore pickle module names after imports.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    inference.INFERENCE_SIZE = (8192, 5642)
    inference.main()
