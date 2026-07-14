#!/usr/bin/env python3
"""Compare AstroNet scores with DR25 Robovetter scores on one held-out set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from astronet_dr25.metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    frame = frame.dropna(subset=["label", "astronet_score", "koi_score"])
    labels = frame["label"].astype(int).to_numpy()
    astro_scores = frame["astronet_score"].to_numpy(float)
    robo_scores = frame["koi_score"].to_numpy(float)
    result = {
        "evaluation_scope": "DR25 label agreement; not independent ground truth",
        "circularity_warning": (
            "koi_pdisposition and koi_score are both DR25 Robovetter products. "
            "These metrics measure agreement/imitation, not scientific superiority."
        ),
        "astronet": binary_metrics(labels, astro_scores),
        "robovetter": binary_metrics(labels, robo_scores),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "astronet_vs_robovetter.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for name, scores in (
        ("AstroNet", astro_scores),
        ("Robovetter", robo_scores),
    ):
        RocCurveDisplay.from_predictions(labels, scores, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(labels, scores, name=name, ax=axes[1])
    astro_pred = (astro_scores >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        labels, astro_pred, display_labels=["FP", "Candidate"], ax=axes[2], colorbar=False
    )
    axes[0].set_title("ROC: same held-out KIC set")
    axes[1].set_title("Precision–recall")
    axes[2].set_title("AstroNet confusion matrix")
    fig.savefig(args.output / "astronet_vs_robovetter.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
