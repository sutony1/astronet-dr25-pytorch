import numpy as np

from astronet_dr25.preprocessing import make_local_global_views


def test_view_shapes_and_normalization() -> None:
    time = np.linspace(0, 100, 100_000, endpoint=False)
    phase = ((time - 0.2 + 1.0) % 2.0) - 1.0
    flux = np.ones_like(time)
    flux[np.abs(phase) < 0.04] -= 0.01
    local, global_view = make_local_global_views(time, flux, 2.0, 0.2, 1.92)
    assert local.shape == (201,)
    assert global_view.shape == (2001,)
    assert np.isclose(np.median(local), 0.0)
    assert np.isclose(np.min(local), -1.0)
    assert np.isclose(np.median(global_view), 0.0)
    assert np.isclose(np.min(global_view), -1.0)
