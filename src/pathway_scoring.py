"""Cross-platform, outcome-free pathway feature transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def compute_rank_pathway_scores(
    expression: pd.DataFrame,
    genesets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Average centered within-sample gene percentiles for each pathway."""
    if expression.empty:
        raise ValueError("Expression matrix must not be empty.")
    if expression.columns.has_duplicates:
        raise ValueError("Expression gene symbols must be unique.")
    numeric = expression.apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Expression matrix contains infinite values.")

    centered_ranks = numeric.rank(axis=1, method="average", pct=True) - 0.5
    available_genes = set(centered_ranks.columns)
    scores: Dict[str, pd.Series] = {}
    for pathway_name, pathway_genes in genesets.items():
        matched_genes = list(
            dict.fromkeys(
                str(gene).strip()
                for gene in pathway_genes
                if str(gene).strip() in available_genes
            )
        )
        if matched_genes:
            scores[pathway_name] = (
                centered_ranks.loc[:, matched_genes]
                .mean(axis=1, skipna=True)
                .fillna(0.0)
            )
        else:
            scores[pathway_name] = pd.Series(0.0, index=numeric.index)
    return pd.DataFrame(scores, index=numeric.index)


@dataclass(frozen=True)
class RobustPathwayScaler:
    medians: pd.Series
    iqrs: pd.Series

    @classmethod
    def fit(cls, training_scores: pd.DataFrame) -> "RobustPathwayScaler":
        if training_scores.empty:
            raise ValueError("Training pathway scores must not be empty.")
        medians = training_scores.median(axis=0)
        iqrs = training_scores.quantile(0.75) - training_scores.quantile(0.25)
        iqrs = iqrs.replace(0.0, 1.0).fillna(1.0)
        return cls(medians=medians.astype(float), iqrs=iqrs.astype(float))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RobustPathwayScaler":
        if payload.get("method") != "training-median-iqr-v1":
            raise ValueError("Unsupported pathway scaler checkpoint contract.")
        medians_raw = payload.get("medians")
        iqrs_raw = payload.get("iqrs")
        if not isinstance(medians_raw, Mapping) or not isinstance(iqrs_raw, Mapping):
            raise ValueError("Pathway scaler checkpoint is missing medians or IQRs.")
        if list(medians_raw) != list(iqrs_raw):
            raise ValueError("Pathway scaler checkpoint columns do not match.")
        medians = pd.Series(medians_raw, dtype=float)
        iqrs = pd.Series(iqrs_raw, dtype=float)
        if (
            not np.isfinite(medians.to_numpy()).all()
            or not np.isfinite(iqrs.to_numpy()).all()
            or (iqrs <= 0.0).any()
        ):
            raise ValueError("Pathway scaler checkpoint contains invalid values.")
        return cls(medians=medians, iqrs=iqrs)

    def transform(self, scores: pd.DataFrame) -> pd.DataFrame:
        expected_columns = self.medians.index.tolist()
        if scores.columns.tolist() != expected_columns:
            raise ValueError("Pathway score columns do not match the training contract.")
        transformed = scores.sub(self.medians, axis=1).div(self.iqrs, axis=1)
        if not np.isfinite(transformed.to_numpy(dtype=float)).all():
            raise ValueError("Scaled pathway scores contain non-finite values.")
        return transformed

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": "training-median-iqr-v1",
            "medians": {key: float(value) for key, value in self.medians.items()},
            "iqrs": {key: float(value) for key, value in self.iqrs.items()},
        }
