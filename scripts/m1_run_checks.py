#!/usr/bin/env python3
"""Run M1 consistency checks without requiring pytest."""

from __future__ import annotations

import json

import numpy as np
import sklearn
import torch

from astronet_dr25.model import AstroNet, OFFICIAL_PARAMETER_COUNT
from astronet_dr25.preprocessing import make_local_global_views


def main() -> None:
    model = AstroNet()
    local = torch.zeros(2, 201)
    global_view = torch.zeros(2, 2001)
    local_shape = tuple(model.local_column(local.unsqueeze(1)).shape)
    global_shape = tuple(model.global_column(global_view.unsqueeze(1)).shape)
    output_shape = tuple(model(local, global_view).shape)
    assert local_shape == (2, 32, 46), local_shape
    assert global_shape == (2, 256, 59), global_shape
    assert output_shape == (2,), output_shape
    assert model.parameter_count == OFFICIAL_PARAMETER_COUNT, model.parameter_count

    time = np.linspace(0, 100, 100_000, endpoint=False)
    phase = ((time - 0.2 + 1.0) % 2.0) - 1.0
    flux = np.ones_like(time)
    flux[np.abs(phase) < 0.04] -= 0.01
    local_result, global_result = make_local_global_views(
        time, flux, 2.0, 0.2, 1.92
    )
    assert local_result.shape == (201,)
    assert global_result.shape == (2001,)
    assert np.isclose(np.median(local_result), 0.0)
    assert np.isclose(np.min(local_result), -1.0)
    assert np.isclose(np.median(global_result), 0.0)
    assert np.isclose(np.min(global_result), -1.0)

    result = {
        "status": "passed",
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "local_hidden_shape": local_shape,
        "global_hidden_shape": global_shape,
        "output_shape": output_shape,
        "parameter_count": model.parameter_count,
        "local_view_normalization": {
            "median": float(np.median(local_result)),
            "minimum": float(np.min(local_result)),
        },
        "global_view_normalization": {
            "median": float(np.median(global_result)),
            "minimum": float(np.min(global_result)),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
