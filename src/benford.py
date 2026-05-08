"""
Utilities for computing Benford-law conformity features from numerical arrays.

This module is designed for BENADV-style experiments where token-level
embedding matrices are flattened into numerical vectors and evaluated against
Benford's law using first significant digits.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import numpy as np


DIGITS = np.arange(1, 10)
BENFORD_PROBS = np.log10(1.0 + 1.0 / DIGITS)


@dataclass(frozen=True)
class BenfordFeatures:
    """Container for Benford conformity metrics of one numeric sample."""

    kl: float
    chi2: float
    mse: float
    r2: float
    n_numbers: int
    n_valid_numbers: int
    digit_1: float
    digit_2: float
    digit_3: float
    digit_4: float
    digit_5: float
    digit_6: float
    digit_7: float
    digit_8: float
    digit_9: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def benford_distribution() -> np.ndarray:
    """Return the theoretical first-digit Benford distribution for digits 1..9."""
    return BENFORD_PROBS.copy()


def first_significant_digits(
    values: np.ndarray | Iterable[float],
    *,
    scale: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Extract first significant digits from a numeric array.

    Parameters
    ----------
    values:
        Numeric values, e.g. flattened embedding values.
    scale:
        Optional positive scaling factor. Benford's law is scale-invariant in
        ideal settings, but the original paper applies a scaling factor before
        extracting significant digits. Use 1.0 unless you intentionally want to
        reproduce a specific scaling choice.
    eps:
        Values with absolute magnitude <= eps are ignored.

    Returns
    -------
    np.ndarray
        Integer array containing digits in {1, ..., 9}.

    Examples
    --------
    >>> first_significant_digits(np.array([-0.00452, 12.7, 0.0, 9.1]))
    array([4, 1, 9])
    """
    if scale <= 0:
        raise ValueError("scale must be positive")

    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = np.abs(arr) * scale

    finite_mask = np.isfinite(arr)
    nonzero_mask = arr > eps
    arr = arr[finite_mask & nonzero_mask]

    if arr.size == 0:
        return np.array([], dtype=np.int64)

    # Move each value into [1, 10) using base-10 order of magnitude.
    magnitudes = np.floor(np.log10(arr))
    significands = arr / np.power(10.0, magnitudes)

    digits = np.floor(significands).astype(np.int64)

    # Numerical guard: values extremely close to 10 can produce digit 10.
    digits = np.clip(digits, 1, 9)
    return digits


def digit_distribution(digits: np.ndarray | Iterable[int]) -> np.ndarray:
    """
    Convert first significant digits into an empirical distribution over 1..9.

    If no valid digits are supplied, returns an all-NaN vector.
    """
    arr = np.asarray(digits, dtype=np.int64).ravel()
    arr = arr[(arr >= 1) & (arr <= 9)]

    if arr.size == 0:
        return np.full(9, np.nan, dtype=np.float64)

    counts = np.bincount(arr, minlength=10)[1:10].astype(np.float64)
    return counts / counts.sum()


def kl_divergence(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute KL(observed || expected) for digit distributions."""
    if expected is None:
        expected = BENFORD_PROBS

    p = np.asarray(observed, dtype=np.float64)
    q = np.asarray(expected, dtype=np.float64)

    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return float("nan")

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def chi_square_statistic(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Compute a probability-scale chi-square distance.

    This follows the common goodness-of-fit form sum((p - q)^2 / q), using
    probabilities rather than raw counts. For formal p-values, raw counts and
    degrees of freedom should be used separately.
    """
    if expected is None:
        expected = BENFORD_PROBS

    p = np.asarray(observed, dtype=np.float64)
    q = np.asarray(expected, dtype=np.float64)

    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return float("nan")

    q = np.clip(q, eps, None)
    return float(np.sum((p - q) ** 2 / q))


def mean_squared_error(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
) -> float:
    """Compute MSE between observed and expected digit distributions."""
    if expected is None:
        expected = BENFORD_PROBS

    p = np.asarray(observed, dtype=np.float64)
    q = np.asarray(expected, dtype=np.float64)

    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return float("nan")

    return float(np.mean((p - q) ** 2))


def r2_score_distribution(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Compute an R^2-style goodness-of-fit score.

    The expected Benford probabilities are treated as predictions of the
    observed empirical probabilities. Values closer to 1 indicate better fit.
    """
    if expected is None:
        expected = BENFORD_PROBS

    p = np.asarray(observed, dtype=np.float64)
    q = np.asarray(expected, dtype=np.float64)

    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return float("nan")

    ss_res = np.sum((p - q) ** 2)
    ss_tot = np.sum((p - np.mean(p)) ** 2)

    if ss_tot <= eps:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def compute_benford_features(
    values: np.ndarray | Iterable[float],
    *,
    scale: float = 1.0,
    eps: float = 1e-12,
) -> BenfordFeatures:
    """
    Compute Benford-law conformity features for a numeric sample.

    Parameters
    ----------
    values:
        Numeric values, usually a flattened embedding matrix.
    scale:
        Optional scaling factor applied before extracting first significant
        digits.
    eps:
        Near-zero values are ignored.

    Returns
    -------
    BenfordFeatures
        KL, chi-square, MSE, R^2, sample-size metadata, and the empirical
        first-digit distribution.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n_numbers = int(arr.size)

    digits = first_significant_digits(arr, scale=scale, eps=eps)
    dist = digit_distribution(digits)

    return BenfordFeatures(
        kl=kl_divergence(dist),
        chi2=chi_square_statistic(dist),
        mse=mean_squared_error(dist),
        r2=r2_score_distribution(dist),
        n_numbers=n_numbers,
        n_valid_numbers=int(digits.size),
        digit_1=float(dist[0]),
        digit_2=float(dist[1]),
        digit_3=float(dist[2]),
        digit_4=float(dist[3]),
        digit_5=float(dist[4]),
        digit_6=float(dist[5]),
        digit_7=float(dist[6]),
        digit_8=float(dist[7]),
        digit_9=float(dist[8]),
    )


def compute_benford_feature_dict(
    values: np.ndarray | Iterable[float],
    *,
    prefix: str = "",
    scale: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Convenience wrapper returning a flat dictionary, optionally prefixed.

    Useful when building pandas rows such as:
    ``row.update(compute_benford_feature_dict(emb, prefix='xlmr_'))``.
    """
    features = compute_benford_features(values, scale=scale, eps=eps).to_dict()

    if prefix:
        return {f"{prefix}{key}": value for key, value in features.items()}

    return features


if __name__ == "__main__":
    # Tiny smoke test.
    rng = np.random.default_rng(42)
    sample = rng.normal(size=(12, 768))
    features = compute_benford_features(sample)
    print(features.to_dict())
