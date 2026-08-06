import importlib
import warnings

import numpy as np
import pandas as pd
import pytest


def _load_module():
    try:
        return importlib.import_module("src.cox_ensemble")
    except ModuleNotFoundError:
        pytest.fail("src.cox_ensemble must implement training-only Cox ensembling")


def test_predict_cox_ensemble_uses_each_training_risk_scale():
    cox_ensemble = _load_module()
    features = pd.DataFrame(
        {"P1": [0.0, 2.0], "P2": [1.0, 1.0]},
        index=["S1", "S2"],
    )
    checkpoint = {
        "model_type": "elastic_net_cox_ensemble_v1",
        "pathway_names": ["P1", "P2"],
        "members": [
            {
                "coefficients": [1.0, 0.0],
                "training_risk_mean": 0.0,
                "training_risk_std": 1.0,
            },
            {
                "coefficients": [3.0, 0.0],
                "training_risk_mean": 1.0,
                "training_risk_std": 2.0,
            },
        ],
    }

    prediction = cox_ensemble.predict_cox_ensemble(features, checkpoint)

    np.testing.assert_allclose(prediction, [-0.25, 2.25])


def test_fit_elastic_net_cox_ensemble_builds_stable_training_only_checkpoint():
    cox_ensemble = _load_module()
    rng = np.random.default_rng(42)
    signal = np.linspace(-2.0, 2.0, 60)
    features = pd.DataFrame(
        {
            "SIGNAL": signal,
            "NOISE-1": rng.normal(size=60),
            "NOISE-2": rng.normal(size=60),
        },
        index=[f"S{index:02d}" for index in range(60)],
    )
    times = pd.Series(100.0 * np.exp(-signal), index=features.index)
    events = pd.Series(([1, 0, 1] * 20), index=features.index)

    checkpoint = cox_ensemble.fit_elastic_net_cox_ensemble(
        features,
        times,
        events,
        penalizer_grid=[0.01],
        l1_ratio_grid=[0.5],
        n_splits=3,
        n_repeats=1,
        seed=42,
        min_selection_frequency=0.5,
    )

    assert checkpoint["model_type"] == "elastic_net_cox_ensemble_v1"
    assert checkpoint["pathway_names"] == features.columns.tolist()
    assert checkpoint["best_penalizer"] == 0.01
    assert checkpoint["best_l1_ratio"] == 0.5
    assert len(checkpoint["members"]) == 3
    assert "SIGNAL" in checkpoint["stable_pathways"]
    predictions = cox_ensemble.predict_cox_ensemble(features, checkpoint)
    assert predictions.shape == (60,)
    assert np.isfinite(predictions).all()


def test_fit_elastic_net_cox_ensemble_skips_failed_hyperparameter_setting(
    monkeypatch,
):
    cox_ensemble = _load_module()
    original_fit = cox_ensemble._fit_coefficients

    def fail_unpenalized(*args, **kwargs):
        penalizer = args[3]
        if penalizer == 0.0:
            raise RuntimeError("singular matrix")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(cox_ensemble, "_fit_coefficients", fail_unpenalized)
    features = pd.DataFrame(
        {
            "P1": np.linspace(-1.0, 1.0, 30),
            "P2": np.linspace(1.0, -1.0, 30),
        }
    )
    times = pd.Series(np.linspace(30.0, 1.0, 30))
    events = pd.Series([1, 0, 1] * 10)

    checkpoint = cox_ensemble.fit_elastic_net_cox_ensemble(
        features,
        times,
        events,
        penalizer_grid=[0.0, 0.1],
        l1_ratio_grid=[0.5],
        n_splits=3,
        n_repeats=1,
        seed=42,
        min_selection_frequency=0.5,
    )

    assert checkpoint["best_penalizer"] == 0.1
    assert len(checkpoint["cv_results"]) == 1
    assert checkpoint["failed_settings"] == [{
        "penalizer": 0.0,
        "l1_ratio": 0.5,
        "error": "singular matrix",
    }]


@pytest.mark.parametrize(
    "warning_name", ["ConvergenceWarning", "LinAlgWarning"]
)
def test_fit_coefficients_rejects_numerical_warning(monkeypatch, warning_name):
    cox_ensemble = _load_module()

    class WarningFitter:
        def fit(self, *args, **kwargs):
            warnings.warn(
                "Newton-Raphson failed",
                getattr(cox_ensemble, warning_name),
            )

    monkeypatch.setattr(
        cox_ensemble,
        "CoxPHFitter",
        lambda **kwargs: WarningFitter(),
    )
    features = pd.DataFrame({"P1": [0.0, 1.0, 2.0]})

    with pytest.raises(getattr(cox_ensemble, warning_name)):
        cox_ensemble._fit_coefficients(
            features,
            pd.Series([3.0, 2.0, 1.0]),
            pd.Series([1, 0, 1]),
            penalizer=0.01,
            l1_ratio=0.5,
        )
