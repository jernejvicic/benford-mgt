"""
Experiment 01: Pure Benford-signal extraction.

This script computes Benford-law conformity features from token-level
transformer embeddings for multilingual human-vs-machine text detection.

Expected input CSV columns by default:
    - text
    - language
    - label

Default label convention:
    - human = 0
    - ai / machine-generated = 1

Example
-------
python src/experiment_01_signal.py \
    --input data/multisocial.csv \
    --output results/features/experiment_01_xlmr.csv \
    --model xlm-roberta-base \
    --languages sl cs sk en \
    --text-col text \
    --language-col language \
    --label-col label
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from benford import compute_benford_feature_dict
from embeddings import TransformerEmbeddingExtractor


LANGUAGE_ALIASES: Dict[str, str] = {
    # English
    "en": "en",
    "eng": "en",
    "english": "en",
    # Slovene / Slovenian
    "sl": "sl",
    "slv": "sl",
    "slovene": "sl",
    "slovenian": "sl",
    # Czech
    "cs": "cs",
    "cz": "cs",
    "ces": "cs",
    "czech": "cs",
    # Slovak
    "sk": "sk",
    "slk": "sk",
    "slo": "sk",
    "slovak": "sk",
}

LABEL_ALIASES: Dict[str, int] = {
    "0": 0,
    "human": 0,
    "hwt": 0,
    "real": 0,
    "authentic": 0,
    "original": 0,
    "1": 1,
    "ai": 1,
    "mgt": 1,
    "machine": 1,
    "machine-generated": 1,
    "machine_generated": 1,
    "generated": 1,
    "synthetic": 1,
    "llm": 1,
}


def normalize_language(value: object) -> Optional[str]:
    """Normalize language labels to ISO-like codes: en, sl, cs, sk."""
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    return LANGUAGE_ALIASES.get(key)


def normalize_label(value: object) -> Optional[int]:
    """Normalize human/AI labels to 0/1."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)

    if isinstance(value, (float, np.floating)) and not np.isnan(value):
        if int(value) in (0, 1) and float(value).is_integer():
            return int(value)

    key = str(value).strip().lower()
    return LABEL_ALIASES.get(key)


def load_and_filter_data(
    input_path: Path,
    *,
    text_col: str,
    language_col: str,
    label_col: str,
    languages: Iterable[str],
    max_samples_per_language_label: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load CSV data, normalize labels/languages, and filter target languages."""
    df = pd.read_csv(input_path)

    required_cols = {text_col, language_col, label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[text_col, language_col, label_col]].copy()
    df = df.rename(
        columns={
            text_col: "text",
            language_col: "language_raw",
            label_col: "label_raw",
        }
    )

    df["language"] = df["language_raw"].map(normalize_language)
    df["label"] = df["label_raw"].map(normalize_label)
    df["text"] = df["text"].astype(str)

    languages = set(languages)
    df = df[df["language"].isin(languages)]
    df = df[df["label"].isin([0, 1])]
    df = df[df["text"].str.strip().str.len() > 0]

    if max_samples_per_language_label is not None:
        df = (
            df.groupby(["language", "label"], group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=min(len(group), max_samples_per_language_label),
                    random_state=random_seed,
                )
            )
            .reset_index(drop=True)
        )

    df = df.reset_index(drop=True)
    df.insert(0, "sample_id", np.arange(len(df)))
    return df


def compute_features_for_dataframe(
    df: pd.DataFrame,
    *,
    model_name: str,
    device: Optional[str],
    max_length: int,
    scale: float,
    include_special_tokens: bool,
    trust_remote_code: bool,
) -> pd.DataFrame:
    """Extract embeddings and compute Benford features for all rows."""
    extractor = TransformerEmbeddingExtractor(
        model_name,
        device=device,
        max_length=max_length,
        include_special_tokens=include_special_tokens,
        trust_remote_code=trust_remote_code,
    )

    rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Computing features"):
        result = extractor.encode(row.text)
        features = compute_benford_feature_dict(result.flatten(), scale=scale)

        output_row = {
            "sample_id": row.sample_id,
            "language": row.language,
            "label": int(row.label),
            "label_name": "ai" if int(row.label) == 1 else "human",
            "model_name": model_name,
            "n_tokens": result.n_tokens,
            "hidden_size": result.hidden_size,
            "n_embedding_numbers": result.n_numbers,
            "text_length_chars": len(row.text),
        }
        output_row.update(features)
        rows.append(output_row)

    return pd.DataFrame(rows)


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create a compact per-language, per-label summary table."""
    metrics = ["kl", "chi2", "mse", "r2", "n_tokens", "n_valid_numbers"]
    summary = (
        features.groupby(["language", "label_name"])[metrics]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Benford features from transformer embeddings.")

    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Output feature CSV path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary CSV path.")

    parser.add_argument("--model", default="xlm-roberta-base", help="HuggingFace model name.")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or omitted for auto.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokenizer length.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor before digit extraction.")
    parser.add_argument("--include-special-tokens", action="store_true", help="Include special tokens in embeddings.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable HuggingFace trust_remote_code.")

    parser.add_argument("--text-col", default="text", help="Input text column name.")
    parser.add_argument("--language-col", default="language", help="Input language column name.")
    parser.add_argument("--label-col", default="label", help="Input label column name.")
    parser.add_argument("--languages", nargs="+", default=["sl", "cs", "sk", "en"], help="Languages to keep.")
    parser.add_argument(
        "--max-samples-per-language-label",
        type=int,
        default=None,
        help="Optional cap per language and label for quick runs.",
    )
    parser.add_argument("--random-seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_and_filter_data(
        args.input,
        text_col=args.text_col,
        language_col=args.language_col,
        label_col=args.label_col,
        languages=args.languages,
        max_samples_per_language_label=args.max_samples_per_language_label,
        random_seed=args.random_seed,
    )

    if df.empty:
        raise ValueError("No rows remain after filtering. Check language and label columns.")

    print("Loaded rows:")
    print(df.groupby(["language", "label"]).size())

    features = compute_features_for_dataframe(
        df,
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        scale=args.scale,
        include_special_tokens=args.include_special_tokens,
        trust_remote_code=args.trust_remote_code,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Saved features to {args.output}")

    summary = summarize_features(features)
    if args.summary_output is None:
        args.summary_output = args.output.with_name(args.output.stem + "_summary.csv")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    print(f"Saved summary to {args.summary_output}")


if __name__ == "__main__":
    main()
