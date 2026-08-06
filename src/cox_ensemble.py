"""Training-only Elastic-Net Cox selection and fold ensembling."""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Tuple
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index
from scipy.linalg import LinAlgWarning
from sklearn.model_selection import RepeatedStratifiedKFold


def _fit_coefficients(
    features: pd.DataFrame,
    times: pd.Series,
    events: pd.Series,
    penalizer: float,
    l1_ratio: float,
) -> np.ndarray:
    training_frame = features.copy()
    training_frame["__time"] = times.to_numpy(dtype=float)
    training_frame["__event"] = events.to_numpy(dtype=int)
    fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", LinAlgWarning)
        fitter.fit(
            training_frame,
            duration_col="__time",
            event_col="__event",
            show_progress=False,
        )
    return fitter.params_.reindex(features.columns).to_numpy(dtype=float)


def _build_splits(
    events: pd.Series,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    indices = np.arange(len(events))
    return list(splitter.split(indices, events.to_numpy(dtype=int)))


def predict_cox_ensemble(features: pd.DataFrame, checkpoint: Dict) -> np.ndarray:
    """Average member log-risk scores after their training-only scaling."""
    if checkpoint.get("model_type") != "elastic_net_cox_ensemble_v1":
        raise ValueError("Unsupported Cox ensemble checkpoint type.")
    pathway_names = [str(value) for value in checkpoint.get("pathway_names", [])]
    if features.columns.tolist() != pathway_names:
        raise ValueError("Evaluation pathways do not match the ensemble checkpoint.")
    members = checkpoint.get("members", [])
    if not members:
        raise ValueError("Cox ensemble checkpoint contains no members.")

    matrix = features.to_numpy(dtype=float)
    member_predictions = []
    for member in members:
        coefficients = np.asarray(member["coefficients"], dtype=float)
        if coefficients.shape != (len(pathway_names),):
            raise ValueError("Cox ensemble member has invalid coefficients.")
        risk_std = float(member["training_risk_std"])
        if not np.isfinite(risk_std) or risk_std <= 0:
            raise ValueError("Cox ensemble member has invalid training risk scale.")
        risk = matrix @ coefficients
        member_predictions.append(
            (risk - float(member["training_risk_mean"])) / risk_std
        )
    prediction = np.mean(np.vstack(member_predictions), axis=0)
    if not np.isfinite(prediction).all():
        raise ValueError("Cox ensemble produced non-finite predictions.")
    return prediction


def fit_elastic_net_cox_ensemble(
    features: pd.DataFrame,
    times: pd.Series,
    events: pd.Series,
    penalizer_grid: Iterable[float],
    l1_ratio_grid: Iterable[float],
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    min_selection_frequency: float = 0.6,
) -> Dict:
    """Tune and fit an ensemble using only the supplied training cohort."""
    if features.empty or features.columns.has_duplicates:
        raise ValueError("Training pathway features must be non-empty and unique.")
    times = pd.Series(times, index=features.index, dtype=float)
    events = pd.Series(events, index=features.index, dtype=int)
    if not events.isin([0, 1]).all():
        raise ValueError("Training event labels must be binary.")
    if not 0 < min_selection_frequency <= 1:
        raise ValueError("min_selection_frequency must be in (0, 1].")

    penalizers = [float(value) for value in penalizer_grid]
    l1_ratios = [float(value) for value in l1_ratio_grid]
    if not penalizers or not l1_ratios:
        raise ValueError("Elastic-Net hyperparameter grids must not be empty.")
    if any(value < 0 for value in penalizers):
        raise ValueError("Elastic-Net penalizers must be non-negative.")
    if any(value < 0 or value > 1 for value in l1_ratios):
        raise ValueError("Elastic-Net l1_ratio values must be in [0, 1].")

    splits = _build_splits(events, n_splits, n_repeats, seed)
    cv_results = []
    failed_settings = []
    fitted_by_setting: Dict[Tuple[float, float], list[np.ndarray]] = {}
    for penalizer, l1_ratio in product(penalizers, l1_ratios):
        fold_scores = []
        fold_coefficients = []
        try:
            for train_index, validation_index in splits:
                coefficients = _fit_coefficients(
                    features.iloc[train_index],
                    times.iloc[train_index],
                    events.iloc[train_index],
                    penalizer,
                    l1_ratio,
                )
                validation_risk = (
                    features.iloc[validation_index].to_numpy() @ coefficients
                )
                fold_scores.append(
                    concordance_index(
                        times.iloc[validation_index],
                        -validation_risk,
                        events.iloc[validation_index],
                    )
                )
                fold_coefficients.append(coefficients)
        except Exception as exc:
            failed_settings.append(
                {
                    "penalizer": penalizer,
                    "l1_ratio": l1_ratio,
                    "error": str(exc),
                }
            )
            continue
        fitted_by_setting[(penalizer, l1_ratio)] = fold_coefficients
        cv_results.append(
            {
                "penalizer": penalizer,
                "l1_ratio": l1_ratio,
                "mean_c_index": float(np.mean(fold_scores)),
                "std_c_index": float(np.std(fold_scores)),
                "fold_c_indices": [float(value) for value in fold_scores],
            }
        )

    if not cv_results:
        raise RuntimeError(
            "All Elastic-Net Cox hyperparameter settings failed to converge."
        )
    best_result = max(
        cv_results,
        key=lambda result: (
            result["mean_c_index"],
            -result["std_c_index"],
            result["penalizer"],
            result["l1_ratio"],
        ),
    )
    best_key = (best_result["penalizer"], best_result["l1_ratio"])
    coefficient_matrix = np.vstack(fitted_by_setting[best_key])
    selection_frequency = np.mean(np.abs(coefficient_matrix) > 1e-7, axis=0)
    stable_mask = selection_frequency >= min_selection_frequency
    if not stable_mask.any():
        stable_mask[np.argmax(np.median(np.abs(coefficient_matrix), axis=0))] = True
    stable_pathways = features.columns[stable_mask].tolist()

    members = []
    for train_index, _ in splits:
        stable_coefficients = _fit_coefficients(
            features.iloc[train_index].loc[:, stable_pathways],
            times.iloc[train_index],
            events.iloc[train_index],
            best_result["penalizer"],
            best_result["l1_ratio"],
        )
        coefficients = np.zeros(features.shape[1], dtype=float)
        coefficients[stable_mask] = stable_coefficients
        training_risk = features.iloc[train_index].to_numpy() @ coefficients
        training_risk_std = float(np.std(training_risk, ddof=1))
        if not np.isfinite(training_risk_std) or training_risk_std <= 0:
            training_risk_std = 1.0
        members.append(
            {
                "coefficients": coefficients.tolist(),
                "training_risk_mean": float(np.mean(training_risk)),
                "training_risk_std": training_risk_std,
            }
        )

    return {
        "model_type": "elastic_net_cox_ensemble_v1",
        "pathway_names": features.columns.tolist(),
        "best_penalizer": best_result["penalizer"],
        "best_l1_ratio": best_result["l1_ratio"],
        "best_cv_c_index": best_result["mean_c_index"],
        "stable_pathways": stable_pathways,
        "selection_frequency": {
            pathway: float(frequency)
            for pathway, frequency in zip(features.columns, selection_frequency)
        },
        "cv_results": cv_results,
        "failed_settings": failed_settings,
        "members": members,
        "training_config": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "seed": seed,
            "min_selection_frequency": min_selection_frequency,
        },
    }
