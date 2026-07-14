#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/mnt/cloud_3t5/exoplanet_data}"
PYTHON="${PYTHON:-${PROJECT}/.venv/bin/python}"

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

INDEX="${TRAINING_INDEX:-${DATA_ROOT}/processed/training_index_dr25_v1_a731be72_7c2d810f_ae453897.parquet}"
PLAN="${SUPERVISED_PLAN:-${DATA_ROOT}/manifests/astronet_dr25_supervised_plan_v1/supervised_lightcurve_plan.csv}"
DATA="${LIGHTCURVE_ROOT:-${DATA_ROOT}/lightcurves_raw/kepler_dr25_supervised_v1}"
M2_MANIFEST="${M2_MANIFEST:-${DATA_ROOT}/manifests/astronet_dr25_m2_1000_v1}"
VIEWS="${VIEW_ROOT:-${DATA_ROOT}/processed/astronet_dr25_m2_1000_v1}"
EXPERIMENT="${EXPERIMENT_ROOT:-${PROJECT}/experiments/m2_1000_single_gpu_v1}"
COMPARISON="${COMPARISON_ROOT:-${PROJECT}/reports/m2_1000_comparison_v1}"
OFFICIAL="${GOOGLE_EXOPLANET_ML_ROOT:-${PROJECT}/references/google_exoplanet_ml/exoplanet-ml}"

if [[ ! -s "$M2_MANIFEST/m2_subset_summary.json" ]]; then
  "${PYTHON}" scripts/m2_prepare_subset.py \
    --index "$INDEX" \
    --plan "$PLAN" \
    --data-root "$DATA" \
    --output-root "$M2_MANIFEST" \
    --targets-per-class 500 \
    --seed 20260714
fi

"${PYTHON}" scripts/m1_build_views.py \
  --events "$M2_MANIFEST/m2_events.parquet" \
  --file-manifest "$M2_MANIFEST/m2_file_manifest.csv" \
  --data-root "$DATA" \
  --official-source-root "$OFFICIAL" \
  --output-root "$VIEWS" \
  --seed 42 \
  --resume

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/m1_train.py \
  --view-root "$VIEWS" \
  --output "$EXPERIMENT" \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 2e-4 \
  --seed 42 \
  --device cuda:0

"${PYTHON}" scripts/m1_compare_robovetter.py \
  --predictions "$EXPERIMENT/test_predictions.csv" \
  --output "$COMPARISON"
