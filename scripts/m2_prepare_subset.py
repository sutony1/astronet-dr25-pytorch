#!/usr/bin/env python3
"""Select a balanced, fully downloaded M2 subset without KIC leakage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LABELS = ("CANDIDATE", "FALSE POSITIVE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    index = pd.read_parquet(args.index)
    events = index[index["koi_pdisposition"].isin(LABELS)].copy()
    plan = pd.read_csv(args.plan)
    expected = plan.groupby("kepid")["filename"].agg(lambda values: set(values.astype(str)))

    actual: dict[int, set[str]] = {}
    actual_bytes: dict[int, int] = {}
    for path in args.data_root.glob("*/*.fits"):
        try:
            kepid = int(path.parent.name)
        except ValueError:
            continue
        if path.stat().st_size <= 0:
            continue
        actual.setdefault(kepid, set()).add(path.name)
        actual_bytes[kepid] = actual_bytes.get(kepid, 0) + path.stat().st_size

    complete = {
        int(kepid)
        for kepid, filenames in expected.items()
        if filenames and filenames.issubset(actual.get(int(kepid), set()))
    }
    target_labels = (
        events.groupby("kepid")["koi_pdisposition"]
        .agg(lambda values: ",".join(sorted(set(values.astype(str)))))
    )

    selected_parts = []
    availability = {}
    for label in LABELS:
        eligible = target_labels[(target_labels == label) & target_labels.index.isin(complete)]
        availability[label] = int(len(eligible))
        if len(eligible) < args.targets_per_class:
            raise RuntimeError(
                f"Only {len(eligible)} complete pure-{label} targets; "
                f"need {args.targets_per_class}"
            )
        selected_parts.append(
            eligible.sample(n=args.targets_per_class, random_state=args.seed).rename("target_label")
        )

    selected = pd.concat(selected_parts).sort_index().reset_index()
    selected_ids = set(selected["kepid"].astype(int))
    subset_events = events[events["kepid"].astype(int).isin(selected_ids)].copy()
    subset_plan = plan[plan["kepid"].astype(int).isin(selected_ids)].copy()
    subset_plan["relative_path"] = subset_plan.apply(
        lambda row: f"{int(row.kepid):09d}/{row.filename}", axis=1
    )
    subset_plan["status"] = "existing"
    subset_plan["actual_bytes"] = subset_plan.apply(
        lambda row: (args.data_root / row.relative_path).stat().st_size, axis=1
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_root / "m2_targets.csv", index=False)
    subset_events.to_parquet(args.output_root / "m2_events.parquet", index=False)
    subset_plan.to_csv(args.output_root / "m2_file_manifest.csv", index=False)

    selected_file_bytes = sum(actual_bytes[kepid] for kepid in selected_ids)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "targets_per_class": args.targets_per_class,
        "targets_selected": int(len(selected)),
        "target_label_counts": selected["target_label"].value_counts().to_dict(),
        "complete_targets_available": len(complete),
        "eligible_pure_target_counts": availability,
        "tce_events_selected": int(len(subset_events)),
        "event_label_counts": subset_events["koi_pdisposition"].value_counts().to_dict(),
        "fits_files_selected": int(len(subset_plan)),
        "fits_bytes_selected": int(selected_file_bytes),
        "missing_selected_files": int(
            sum(
                not (args.data_root / relative).is_file()
                for relative in subset_plan["relative_path"]
            )
        ),
        "mixed_kic_excluded": int((target_labels.str.contains(",")).sum()),
    }
    (args.output_root / "m2_subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
