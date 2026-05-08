"""
Analyze Experiment 01: Pure Benford-signal separation.

This script reads the feature CSV produced by experiment_01_signal.py and
compares Benford conformity metrics between human-written and AI-generated text
for each language.

It produces:
    - a statistical test table CSV
    - boxplots for KL, chi2, MSE, and R2

Example
-------
python src/analyze_01_signal.py \
    --features results/features/experiment_01_xlmr.csv \
    --output-table results/tables/experiment_01_tests.csv \
    --plot-dir results/plots/experiment_01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


DEFAULT_METRICS = ["kl", "chi2", "mse", "r2"]


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Compute Cliff's delta effect size.

    Positive values mean x tends to be larger than y.
    Negative values mean x tends to be smaller than y.
    """
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return float("nan")

    greater = 0
    lesser = 0

    # Chunked implementation to avoid creating a huge pairwise matrix.
    for value in x:
        greater += np.sum(value > y)
        lesser += np.sum(value < y)

    return float((greater - lesser) / (x.size * y.size))


def compare_groups(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Run human-vs-AI statistical comparisons per language and metric."""
    rows = []

    for language, group in features.groupby("language"):
        human = group[group["label"] == 0]
        ai = group[group["label"] == 1]

        for metric in metrics:
            human_values = human[metric].dropna().astype(float).to_numpy()
            ai_values = ai[metric].dropna().astype(float).to_numpy()

            if len(human_values) < 2 or len(ai_values) < 2:
                rows.append(
                    {
                        "language": language,
                        "metric": metric,
                        "n_human": len(human_values),
                        "n_ai": len(ai_values),
                        "mean_human": np.nan,
                        "mean_ai": np.nan,
                        "median_human": np.nan,
                        "median_ai": np.nan,
                        "ai_minus_human_mean": np.nan,
                        "expected_direction_holds": np.nan,
                        "t_stat": np.nan,
                        "t_pvalue": np.nan,
                        "mw_stat": np.nan,
                        "mw_pvalue": np.nan,
                        "cliffs_delta_ai_vs_human": np.nan,
                    }
                )
                continue

            t_stat, t_pvalue = ttest_ind(ai_values, human_values, equal_var=False, nan_policy="omit")
            mw_stat, mw_pvalue = mannwhitneyu(ai_values, human_values, alternative="two-sided")

            ai_mean = float(np.mean(ai_values))
            human_mean = float(np.mean(human_values))
            delta = ai_mean - human_mean

            # For distance metrics, AI should be lower if it conforms more closely.
            # For R2, AI should be higher.
            if metric == "r2":
                expected_direction_holds = delta > 0
            else:
                expected_direction_holds = delta < 0

            rows.append(
                {
                    "language": language,
                    "metric": metric,
                    "n_human": len(human_values),
                    "n_ai": len(ai_values),
                    "mean_human": human_mean,
                    "mean_ai": ai_mean,
                    "median_human": float(np.median(human_values)),
                    "median_ai": float(np.median(ai_values)),
                    "ai_minus_human_mean": delta,
                    "expected_direction_holds": bool(expected_direction_holds),
                    "t_stat": float(t_stat),
                    "t_pvalue": float(t_pvalue),
                    "mw_stat": float(mw_stat),
                    "mw_pvalue": float(mw_pvalue),
                    "cliffs_delta_ai_vs_human": cliffs_delta(ai_values, human_values),
                }
            )

    return pd.DataFrame(rows)


def plot_metric_boxplot(features: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Create one boxplot per metric, grouped by language and label."""
    languages = sorted(features["language"].dropna().unique())

    data = []
    positions = []
    labels = []
    position = 1

    for language in languages:
        for label_value, label_name in [(0, "human"), (1, "ai")]:
            values = features[
                (features["language"] == language) & (features["label"] == label_value)
            ][metric].dropna().astype(float).to_numpy()
            data.append(values)
            positions.append(position)
            labels.append(f"{language}\n{label_name}")
            position += 1
        position += 0.5

    plt.figure(figsize=(max(8, len(data) * 0.75), 5))
    plt.boxplot(data, positions=positions, showfliers=False)
    plt.xticks(positions, labels, rotation=0)
    plt.ylabel(metric)
    plt.title(f"Benford conformity metric: {metric}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_histograms(features: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Create overlaid human/AI histograms for each language in separate figures."""
    languages = sorted(features["language"].dropna().unique())

    for language in languages:
        subset = features[features["language"] == language]
        human_values = subset[subset["label"] == 0][metric].dropna().astype(float).to_numpy()
        ai_values = subset[subset["label"] == 1][metric].dropna().astype(float).to_numpy()

        if human_values.size == 0 or ai_values.size == 0:
            continue

        plt.figure(figsize=(7, 4))
        plt.hist(human_values, bins=30, alpha=0.55, label="human", density=True)
        plt.hist(ai_values, bins=30, alpha=0.55, label="ai", density=True)
        plt.xlabel(metric)
        plt.ylabel("density")
        plt.title(f"{metric} distribution: {language}")
        plt.legend()
        plt.tight_layout()

        language_output = output_path.with_name(f"{output_path.stem}_{language}{output_path.suffix}")
        language_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(language_output, dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Benford feature separation between human and AI text.")
    parser.add_argument("--features", type=Path, required=True, help="Feature CSV from experiment_01_signal.py.")
    parser.add_argument("--output-table", type=Path, required=True, help="Output statistical test CSV.")
    parser.add_argument("--plot-dir", type=Path, required=True, help="Directory for plots.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to analyze.")
    parser.add_argument("--histograms", action="store_true", help="Also generate per-language histograms.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features = pd.read_csv(args.features)
    required = {"language", "label", *args.metrics}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tests = compare_groups(features, args.metrics)

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    tests.to_csv(args.output_table, index=False)
    print(f"Saved statistical tests to {args.output_table}")

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in args.metrics:
        plot_metric_boxplot(features, metric, args.plot_dir / f"boxplot_{metric}.png")
        if args.histograms:
            plot_metric_histograms(features, metric, args.plot_dir / f"hist_{metric}.png")

    print(f"Saved plots to {args.plot_dir}")

    print("\nCompact result preview:")
    preview_cols = [
        "language",
        "metric",
        "mean_human",
        "mean_ai",
        "ai_minus_human_mean",
        "expected_direction_holds",
        "mw_pvalue",
        "cliffs_delta_ai_vs_human",
    ]
    print(tests[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
