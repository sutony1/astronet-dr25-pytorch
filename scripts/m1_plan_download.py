#!/usr/bin/env python3
"""Create a resumable, no-download MAST manifest for labeled DR25 targets."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://archive.stsci.edu/pub/kepler/lightcurves"
thread_local = threading.local()


def session_for_thread(workers: int) -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        retry = Retry(
            total=6,
            connect=6,
            read=6,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers)
        session = requests.Session()
        session.mount("https://", adapter)
        session.headers["User-Agent"] = "astronet-dr25-capacity-planner/1.0"
        thread_local.session = session
    return session


def parse_apache_size(token: str) -> tuple[int, int]:
    """Return rounded center and guaranteed rounding upper bound in bytes."""
    match = re.fullmatch(r"([0-9.]+)([KMG]?)", token, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported Apache size token: {token}")
    value = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[suffix]
    decimals = len(match.group(1).partition(".")[2])
    rounding_half_step = 0.5 * 10 ** (-decimals)
    return round(value * multiplier), int((value + rounding_half_step) * multiplier)


def list_target(kepid: int, workers: int) -> list[dict[str, object]]:
    kic = f"{kepid:09d}"
    url = f"{BASE_URL}/{kic[:4]}/{kic}/"
    response = session_for_thread(workers).get(url, timeout=(15, 90))
    response.raise_for_status()
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r'href="([^"]+_llc\.fits)"[^\n]*?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+([0-9.]+[KMG]?)\s+',
        flags=re.IGNORECASE,
    )
    for filename, size_token in pattern.findall(response.text):
        filename = html.unescape(filename)
        center, upper = parse_apache_size(size_token)
        rows.append(
            {
                "kepid": kepid,
                "filename": filename,
                "url": url + filename,
                "mast_size_token": size_token,
                "estimated_bytes": center,
                "rounding_upper_bytes": upper,
            }
        )
    if not rows:
        raise RuntimeError(f"No long-cadence FITS found at {url}")
    return rows


def target_label(values: pd.Series) -> str:
    unique = set(values.dropna().astype(str))
    if unique == {"CANDIDATE"}:
        return "CANDIDATE"
    if unique == {"FALSE POSITIVE"}:
        return "FALSE POSITIVE"
    return "MIXED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--hard-limit-gib", type=float, default=200.0)
    parser.add_argument("--limit-targets", type=int)
    args = parser.parse_args()
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    index = pd.read_parquet(args.index)
    labeled = index[index["koi_pdisposition"].isin(["CANDIDATE", "FALSE POSITIVE"])].copy()
    targets = (
        labeled.groupby("kepid")["koi_pdisposition"]
        .apply(target_label)
        .rename("target_label")
        .reset_index()
        .sort_values("kepid")
    )
    if args.limit_targets:
        targets = targets.head(args.limit_targets)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_root / "listing_cache"
    cache_root.mkdir(exist_ok=True)
    targets.to_csv(args.output_root / "supervised_targets.csv", index=False)

    all_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    pending: list[int] = []
    for kepid in targets["kepid"].astype(int):
        cache = cache_root / f"{kepid:09d}.json"
        if cache.exists():
            all_rows.extend(json.loads(cache.read_text(encoding="utf-8")))
        else:
            pending.append(kepid)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(list_target, kepid, args.workers): kepid for kepid in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            kepid = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                (cache_root / f"{kepid:09d}.json").write_text(
                    json.dumps(rows, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as exc:
                errors.append({"kepid": kepid, "error": repr(exc)})
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"listed={completed}/{len(futures)} files={len(all_rows)} errors={len(errors)}",
                    flush=True,
                )

    manifest = pd.DataFrame(all_rows).sort_values(["kepid", "filename"])
    manifest.to_csv(args.output_root / "supervised_lightcurve_plan.csv", index=False)
    hard_limit_bytes = int(args.hard_limit_gib * 1024**3)
    upper_bytes = int(manifest["rounding_upper_bytes"].sum()) if len(manifest) else 0
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "download_started": False,
        "labeled_tces": int(len(labeled)),
        "requested_targets": int(len(targets)),
        "targets_listed": int(manifest["kepid"].nunique()) if len(manifest) else 0,
        "listing_errors": errors,
        "fits_files": int(len(manifest)),
        "mast_rounded_estimated_bytes": int(manifest["estimated_bytes"].sum()) if len(manifest) else 0,
        "rounding_safe_upper_bytes": upper_bytes,
        "hard_limit_bytes": hard_limit_bytes,
        "safe_to_download": not errors and upper_bytes <= hard_limit_bytes,
        "note": "MAST directory sizes are rounded; rounding_safe_upper_bytes adds half a displayed unit per file.",
    }
    (args.output_root / "download_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
