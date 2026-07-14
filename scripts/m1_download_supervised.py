#!/usr/bin/env python3
"""Hard-gated, resumable downloader for the approved DR25 FITS plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


thread_local = threading.local()


def get_session(workers: int) -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        retry = Retry(
            total=8,
            connect=8,
            read=8,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers)
        session = requests.Session()
        session.mount("https://", adapter)
        session.headers["User-Agent"] = "astronet-dr25-supervised-downloader/1.0"
        thread_local.session = session
    return session


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fits(path: Path) -> None:
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty file: {path}")
    with path.open("rb") as handle:
        if not handle.read(80).startswith(b"SIMPLE"):
            raise RuntimeError(f"Not a FITS primary header: {path}")


def fetch(job: dict[str, object], output_root: Path, reuse_root: Path | None, workers: int):
    kic = f"{int(job['kepid']):09d}"
    filename = str(job["filename"])
    destination = output_root / kic / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if destination.exists():
        validate_fits(destination)
    else:
        reusable = reuse_root / kic / filename if reuse_root else None
        if reusable is not None and reusable.exists():
            validate_fits(reusable)
            os.link(reusable, destination)
            status = "hardlinked_pilot"
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                temporary.unlink()
            digest = hashlib.sha256()
            with get_session(workers).get(str(job["url"]), stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
            validate_fits(temporary)
            temporary.replace(destination)
            status = "downloaded"
            return {
                "kepid": int(job["kepid"]),
                "filename": filename,
                "status": status,
                "bytes": destination.stat().st_size,
                "sha256": digest.hexdigest(),
                "path": str(destination),
                "error": "",
            }
    return {
        "kepid": int(job["kepid"]),
        "filename": filename,
        "status": status,
        "bytes": destination.stat().st_size,
        "sha256": sha256_path(destination),
        "path": str(destination),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hard-limit-gib", type=float, default=200.0)
    parser.add_argument("--reserve-gib", type=float, default=50.0)
    parser.add_argument("--limit-files", type=int)
    args = parser.parse_args()
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    plan = pd.read_csv(args.plan)
    if args.limit_files:
        plan = plan.head(args.limit_files)
    planned_upper = int(plan["rounding_upper_bytes"].sum())
    hard_limit = int(args.hard_limit_gib * 1024**3)
    reserve = int(args.reserve_gib * 1024**3)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(args.output_root).free
    if planned_upper > hard_limit:
        raise SystemExit(f"REFUSED: plan upper bound {planned_upper} exceeds hard limit {hard_limit}")
    if free_before < planned_upper + reserve:
        raise SystemExit(
            f"REFUSED: free space {free_before} is below plan+reserve {planned_upper + reserve}"
        )

    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch, row._asdict(), args.output_root, args.reuse_root, args.workers): row
            for row in plan.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "kepid": int(row.kepid),
                        "filename": row.filename,
                        "status": "failed",
                        "bytes": 0,
                        "sha256": "",
                        "path": "",
                        "error": repr(exc),
                    }
                )
            if completed % 200 == 0 or completed == len(futures):
                actual = sum(int(item["bytes"]) for item in results)
                if actual > hard_limit:
                    raise RuntimeError("Hard byte limit exceeded; refusing to continue")
                print(
                    f"completed={completed}/{len(futures)} ok={len(results)} failed={len(errors)} bytes={actual}",
                    flush=True,
                )

    manifest = pd.DataFrame(results + errors).sort_values(["kepid", "filename"])
    manifest.to_csv(args.report_root / "download_manifest.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "planned_files": int(len(plan)),
        "planned_rounding_upper_bytes": planned_upper,
        "hard_limit_bytes": hard_limit,
        "free_bytes_before": free_before,
        "successful_files": len(results),
        "failed_files": len(errors),
        "downloaded_files": sum(item["status"] == "downloaded" for item in results),
        "hardlinked_pilot_files": sum(item["status"] == "hardlinked_pilot" for item in results),
        "preexisting_files": sum(item["status"] == "existing" for item in results),
        "actual_bytes": sum(int(item["bytes"]) for item in results),
        "errors": errors,
    }
    (args.report_root / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
