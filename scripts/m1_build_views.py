#!/usr/bin/env python3
"""Build versioned AstroNet local/global inputs from Kepler FITS files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from astronet_dr25.dataset import grouped_five_fold_split
from astronet_dr25.preprocessing import (
    PREPROCESSING_VERSION,
    flatten_with_official_spline,
    make_local_global_views,
    read_kepler_fits,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--file-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-targets", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    events = pd.read_parquet(args.events)
    events = events[events["koi_pdisposition"].isin(["CANDIDATE", "FALSE POSITIVE"])].copy()
    events["label"] = (events["koi_pdisposition"] == "CANDIDATE").astype("int8")
    files = pd.read_csv(args.file_manifest)
    if "status" in files:
        files = files[files["status"].isin(["downloaded", "existing"])]
    available = sorted(set(events["kepid"].astype(int)) & set(files["kepid"].astype(int)))
    if args.limit_targets:
        available = available[: args.limit_targets]
    events = events[events["kepid"].astype(int).isin(available)].copy()

    views_root = args.output_root / "views"
    views_root.mkdir(parents=True, exist_ok=True)
    existing_index_path = args.output_root / "view_index.parquet"
    partial_index_path = args.output_root / "view_index.partial.parquet"
    resume_index_path = partial_index_path if partial_index_path.exists() else existing_index_path
    if args.resume and resume_index_path.exists():
        existing = pd.read_parquet(resume_index_path)
        rows: list[dict[str, object]] = existing.to_dict(orient="records")
        processed_kic = set(existing["kepid"].astype(int))
        available = [kepid for kepid in available if kepid not in processed_kic]
        print(f"resume_existing_views={len(rows)} remaining_targets={len(available)}", flush=True)
    else:
        rows = []
    failures: list[dict[str, object]] = []

    for completed, kepid in enumerate(available, start=1):
        target_files = files[files["kepid"].astype(int) == kepid]
        paths = [args.data_root / value for value in target_files["relative_path"].tolist()]
        try:
            all_time, all_flux = read_kepler_fits(paths)
            time, flux, spline_metadata = flatten_with_official_spline(
                all_time, all_flux, args.official_source_root
            )
            for event in events[events["kepid"].astype(int) == kepid].itertuples(index=False):
                local_view, global_view = make_local_global_views(
                    time,
                    flux,
                    float(event.tce_period),
                    float(event.tce_time0bk),
                    float(event.tce_duration),
                )
                filename = f"kic_{kepid:09d}_tce_{int(event.tce_plnt_num):02d}.npz"
                output_path = views_root / filename
                np.savez_compressed(
                    output_path,
                    local_view=local_view,
                    global_view=global_view,
                    preprocessing_version=PREPROCESSING_VERSION,
                )
                rows.append(
                    {
                        "row_id": len(rows),
                        "kepid": kepid,
                        "tce_plnt_num": int(event.tce_plnt_num),
                        "label": int(event.label),
                        "koi_pdisposition": event.koi_pdisposition,
                        "koi_disposition": event.koi_disposition,
                        "koi_score": event.koi_score,
                        "view_path": str(output_path.relative_to(args.output_root)),
                        "preprocessing_version": PREPROCESSING_VERSION,
                        **spline_metadata,
                    }
                )
        except Exception as exc:
            failures.append({"kepid": kepid, "error": repr(exc)})
        if completed % 5 == 0 or completed == len(available):
            pd.DataFrame(rows).to_parquet(partial_index_path, index=False)
            (args.output_root / "build_failures.partial.json").write_text(
                json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"targets={completed}/{len(available)} views={len(rows)} failures={len(failures)}",
                flush=True,
            )

    index = pd.DataFrame(rows)
    index["row_id"] = np.arange(len(index), dtype=int)
    if index.empty:
        raise RuntimeError("No views were generated")
    index = grouped_five_fold_split(index, seed=args.seed)
    index.to_parquet(args.output_root / "view_index.parquet", index=False)
    index.to_csv(args.output_root / "view_index.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "targets_requested": len(available),
        "targets_failed": len(failures),
        "views_generated": len(index),
        "split_counts": index["split"].value_counts().to_dict(),
        "split_label_counts": {
            f"{split_name}:{label}": int(count)
            for (split_name, label), count in index.groupby(["split", "label"]).size().items()
        },
        "kic_leakage_max_splits": int(index.groupby("kepid")["split"].nunique().max()),
        "failures": failures,
    }
    (args.output_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial_index_path.unlink(missing_ok=True)
    (args.output_root / "build_failures.partial.json").unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
