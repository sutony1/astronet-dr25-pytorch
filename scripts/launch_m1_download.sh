#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/mnt/cloud_3t5/exoplanet_data}"
PYTHON="${PYTHON:-${PROJECT}/.venv/bin/python}"

PLAN="${DATA_ROOT}/manifests/astronet_dr25_supervised_plan_v1/supervised_lightcurve_plan.csv"
OUTPUT="${DATA_ROOT}/lightcurves_raw/kepler_dr25_supervised_v1"
REUSE_ROOT="${DATA_ROOT}/lightcurves_raw/kepler_dr25_pilot100"
REPORT=${PROJECT}/reports/m1/download_full
LOG=${PROJECT}/reports/m1/download_full.log
PIDFILE=${PROJECT}/reports/m1/download_full.pid

mkdir -p "${REPORT}"
if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
  echo "already_running pid=$(cat "${PIDFILE}")"
  exit 0
fi

cd "${PROJECT}"
nohup "${PYTHON}" scripts/m1_download_supervised.py \
  --plan "${PLAN}" \
  --output-root "${OUTPUT}" \
  --reuse-root "${REUSE_ROOT}" \
  --report-root "${REPORT}" \
  --workers 8 \
  --hard-limit-gib 200 \
  --reserve-gib 50 \
  >"${LOG}" 2>&1 </dev/null &
pid=$!
echo "${pid}" >"${PIDFILE}"
echo "started pid=${pid} log=${LOG}"
