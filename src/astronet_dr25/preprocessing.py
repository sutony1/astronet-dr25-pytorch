"""TensorFlow-free port of the official AstroNet light-curve views."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits


PREPROCESSING_VERSION = "google_astronet_2018_port_dr25_v1"


def read_kepler_fits(paths: Sequence[Path]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read PDCSAP_FLUX like the official kepler_io implementation.

    The original implementation removes non-finite samples but does not apply
    a SAP_QUALITY mask.  We preserve that behavior for baseline fidelity.
    """
    all_time: list[np.ndarray] = []
    all_flux: list[np.ndarray] = []
    for path in sorted(paths):
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            time = np.asarray(data["TIME"], dtype=np.float64)
            flux = np.asarray(data["PDCSAP_FLUX"], dtype=np.float64)
        finite = np.isfinite(time) & np.isfinite(flux)
        if finite.any():
            all_time.append(time[finite])
            all_flux.append(flux[finite])
    if not all_time:
        raise RuntimeError("No finite TIME/PDCSAP_FLUX samples found")
    return all_time, all_flux


def split_on_gaps(
    all_time: Sequence[np.ndarray],
    all_flux: Sequence[np.ndarray],
    gap_width: float = 0.75,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    out_time: list[np.ndarray] = []
    out_flux: list[np.ndarray] = []
    for time, flux in zip(all_time, all_flux):
        boundaries = np.flatnonzero(np.diff(time) > gap_width) + 1
        for t_piece, f_piece in zip(np.split(time, boundaries), np.split(flux, boundaries)):
            if len(t_piece):
                out_time.append(t_piece)
                out_flux.append(f_piece)
    return out_time, out_flux


def flatten_with_official_spline(
    all_time: Sequence[np.ndarray],
    all_flux: Sequence[np.ndarray],
    official_source_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None]]:
    """Use the pinned Google kepler_spline code without importing TensorFlow."""
    source = str(official_source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    # Compatibility aliases for the 2018 code on NumPy 2.x.
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]
    from third_party.kepler_spline import kepler_spline  # type: ignore

    pieces_time, pieces_flux = split_on_gaps(all_time, all_flux, gap_width=0.75)
    spline_method = "bic_20_bkspaces"
    try:
        spline, metadata = kepler_spline.fit_kepler_spline(
            pieces_time, pieces_flux, verbose=False
        )
    except (ValueError, TypeError, IndexError) as exc:
        # pydl 1.0 can raise a shape-broadcast ValueError for a candidate
        # breakpoint that the 2018 wrapper did not anticipate.  Preserve the
        # official spline algorithm, but fall back to its documented 1.5-day
        # default instead of dropping the target.
        spline_method = f"fixed_1.5d_compat_fallback:{type(exc).__name__}"
        spline = []
        for time_piece, flux_piece in zip(pieces_time, pieces_flux):
            try:
                spline_piece, _ = kepler_spline.kepler_spline(
                    time_piece, flux_piece, bkspace=1.5
                )
            except Exception:
                spline_piece = np.full(len(flux_piece), np.nan)
            spline.append(spline_piece)
        metadata = None
    time = np.concatenate(pieces_time)
    flux = np.concatenate(pieces_flux)
    spline_values = np.concatenate(spline)
    finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(spline_values)
    finite &= spline_values != 0
    if finite.sum() < 2:
        raise RuntimeError("Official spline produced fewer than two valid samples")
    normalized = flux[finite] / spline_values[finite]
    return time[finite], normalized, {
        "spline_method": spline_method,
        "spline_bkspace": (
            None if metadata is None or metadata.bkspace is None else float(metadata.bkspace)
        ),
        "spline_bic": None if metadata is None or metadata.bic is None else float(metadata.bic),
    }


def phase_fold_and_sort(
    time: np.ndarray, values: np.ndarray, period: float, t0: float
) -> tuple[np.ndarray, np.ndarray]:
    half_period = period / 2.0
    folded = np.mod(time + (half_period - t0), period) - half_period
    order = np.argsort(folded)
    return folded[order], values[order]


def bin_and_aggregate(
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int,
    bin_width: float,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Modern NumPy port of google exoplanet-ml/light_curve/binning.py."""
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("x and y must have matching lengths of at least two")
    if not x_min < x_max or not 0 < bin_width < (x_max - x_min):
        raise ValueError("Invalid bin interval or width")
    spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    result = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    left_indices = np.searchsorted(x, x_min + spacing * np.arange(num_bins), side="left")
    right_indices = np.searchsorted(
        x, x_min + bin_width + spacing * np.arange(num_bins), side="left"
    )
    for i, (left, right) in enumerate(zip(left_indices, right_indices)):
        if right > left:
            result[i] = np.median(y[left:right])
            counts[i] = right - left
    return result, counts


def generate_view(
    folded_time: np.ndarray,
    values: np.ndarray,
    num_bins: int,
    bin_width: float,
    t_min: float,
    t_max: float,
) -> np.ndarray:
    view, counts = bin_and_aggregate(
        folded_time, values, num_bins, bin_width, t_min, t_max
    )
    view = np.where(counts > 0, view, np.median(values))
    view -= np.median(view)
    depth_scale = abs(float(np.min(view)))
    if not np.isfinite(depth_scale) or depth_scale == 0:
        raise RuntimeError("Cannot normalize a view with zero/invalid minimum")
    view /= depth_scale
    return view.astype(np.float32)


def make_local_global_views(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    epoch_bkjd: float,
    duration_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    if period_days <= 0 or duration_hours <= 0:
        raise ValueError("Period and duration must be positive")
    duration_days = duration_hours / 24.0
    folded_time, folded_flux = phase_fold_and_sort(
        time, flux, period_days, epoch_bkjd
    )
    global_view = generate_view(
        folded_time,
        folded_flux,
        num_bins=2001,
        bin_width=period_days / 2001.0,
        t_min=-period_days / 2.0,
        t_max=period_days / 2.0,
    )
    local_min = max(-period_days / 2.0, -4.0 * duration_days)
    local_max = min(period_days / 2.0, 4.0 * duration_days)
    local_view = generate_view(
        folded_time,
        folded_flux,
        num_bins=201,
        bin_width=0.16 * duration_days,
        t_min=local_min,
        t_max=local_max,
    )
    return local_view, global_view
