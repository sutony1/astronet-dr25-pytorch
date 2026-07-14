#!/usr/bin/env python3
"""Audit DR25 target counts and estimate FITS download storage.

This script is intentionally read-only.  It uses the already downloaded
pilot sample as an empirical basis and writes a machine-readable budget that
can be consumed by the future downloader as a hard safety gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(value)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def pick_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"None of {candidates!r} found in columns: {columns}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--hard-limit-gib",
        type=float,
        default=200.0,
        help="Downloader safety ceiling; no download is performed here.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.index)
    columns = list(frame.columns)
    kic_col = pick_column(columns, ("kepid", "kic", "KIC"))
    disposition_col = next(
        (c for c in ("koi_disposition", "label", "disposition") if c in columns),
        None,
    )

    fits_files = list(args.pilot_root.rglob("*.fits"))
    if not fits_files:
        raise RuntimeError(f"No FITS files found below {args.pilot_root}")
    sizes = pd.Series([p.stat().st_size for p in fits_files], dtype="int64")
    pilot_targets = len({p.parent.name for p in fits_files})
    if not pilot_targets:
        raise RuntimeError("Could not infer pilot target count")

    unique_targets = int(frame[kic_col].nunique())
    supervised_label_col = next(
        (c for c in ("koi_pdisposition", "koi_disposition", "label") if c in columns),
        None,
    )
    if supervised_label_col:
        supervised_mask = frame[supervised_label_col].isin(
            ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE", 0, 1]
        )
        supervised_frame = frame.loc[supervised_mask]
    else:
        supervised_frame = frame.iloc[0:0]
    supervised_rows = int(len(supervised_frame))
    supervised_targets = int(supervised_frame[kic_col].nunique())
    mean_files_per_target = len(fits_files) / pilot_targets
    mean_bytes_per_target = int(sizes.sum() / pilot_targets)

    central_bytes = unique_targets * mean_bytes_per_target
    # The upper estimate protects against a target mix with more/larger quarters.
    conservative_bytes = math.ceil(central_bytes * 1.35)
    hard_limit_bytes = int(args.hard_limit_gib * 1024**3)
    supervised_central_bytes = supervised_targets * mean_bytes_per_target
    supervised_conservative_bytes = math.ceil(supervised_central_bytes * 1.35)

    result = {
        "index_path": str(args.index),
        "index_rows": int(len(frame)),
        "index_columns": columns,
        "unique_kic_targets": unique_targets,
        "supervised": {
            "label_column": supervised_label_col,
            "rows": supervised_rows,
            "unique_kic_targets": supervised_targets,
            "central_bytes": int(supervised_central_bytes),
            "conservative_bytes_1_35x": int(supervised_conservative_bytes),
            "conservative_estimate_within_limit": (
                supervised_conservative_bytes <= hard_limit_bytes
            ),
        },
        "kic_column": kic_col,
        "label_counts": (
            {str(k): int(v) for k, v in frame[disposition_col].value_counts(dropna=False).items()}
            if disposition_col
            else {}
        ),
        "pilot": {
            "targets": pilot_targets,
            "fits_files": len(fits_files),
            "bytes": int(sizes.sum()),
            "mean_files_per_target": mean_files_per_target,
            "mean_bytes_per_target": mean_bytes_per_target,
            "min_file_bytes": int(sizes.min()),
            "median_file_bytes": int(sizes.median()),
            "max_file_bytes": int(sizes.max()),
        },
        "estimate": {
            "central_bytes": int(central_bytes),
            "conservative_bytes_1_35x": int(conservative_bytes),
            "hard_limit_bytes": hard_limit_bytes,
            "hard_limit_gib": args.hard_limit_gib,
            "conservative_estimate_within_limit": conservative_bytes <= hard_limit_bytes,
        },
        "download_started": False,
    }

    audit_keywords = ("label", "train", "disp", "score", "flag")
    result["candidate_label_columns"] = {}
    for column in columns:
        if any(keyword in column.lower() for keyword in audit_keywords):
            counts = frame[column].value_counts(dropna=False).head(20)
            result["candidate_label_columns"][column] = {
                str(key): int(value) for key, value in counts.items()
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    label_rows = "\n".join(
        f"- `{label}`: {count:,}" for label, count in result["label_counts"].items()
    ) or "- 训练索引中没有识别到标签列。"
    markdown = f"""# DR25 AstroNet 下载容量预算

> 本报告只审计和估算，**没有启动全量下载**。

## 精确计数

- 训练索引行数（TCE）：{len(frame):,}
- 唯一 KIC 恒星数：{unique_targets:,}
- 有监督标签行数：{supervised_rows:,}
- 有监督训练所需唯一 KIC：{supervised_targets:,}
- 首批实测目标数：{pilot_targets:,}
- 首批实测 FITS 文件数：{len(fits_files):,}
- 首批实测总量：{human_bytes(sizes.sum())}（{int(sizes.sum()):,} bytes）
- 平均每颗目标：{mean_files_per_target:.2f} 个文件，{human_bytes(mean_bytes_per_target)}

## 标签构成

{label_rows}

## 全量下载估算

- **推荐的有标签训练集中心估计：{human_bytes(supervised_central_bytes)}**
- **推荐的有标签训练集保守估计：{human_bytes(supervised_conservative_bytes)}**
- 全部 TCE 恒星中心估计：{human_bytes(central_bytes)}
- 全部 TCE 恒星保守估计（中心值 × 1.35）：{human_bytes(conservative_bytes)}
- 下载硬上限：{human_bytes(hard_limit_bytes)}
- 保守估计是否低于硬上限：**{conservative_bytes <= hard_limit_bytes}**

实际下载器必须先取得文件级 URL/Content-Length 清单，再以精确字节数复核；只有低于硬上限才允许下载。
"""
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
