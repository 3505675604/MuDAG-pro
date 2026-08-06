import importlib

import numpy as np
import pandas as pd
import pytest


def _load_module():
    try:
        return importlib.import_module("src.pathway_scoring")
    except ModuleNotFoundError:
        pytest.fail("src.pathway_scoring must implement the shared feature contract")


def test_rank_pathway_scores_are_invariant_to_monotonic_platform_transform():
    pathway_scoring = _load_module()
    expression = pd.DataFrame(
        {
            "G1": [1.0, 8.0],
            "G2": [2.0, 4.0],
            "G3": [4.0, 2.0],
            "G4": [8.0, 1.0],
        },
        index=["S1", "S2"],
    )
    genesets = {"LOW": ["G1", "G2"], "HIGH": ["G3", "G4"]}

    scores = pathway_scoring.compute_rank_pathway_scores(expression, genesets)
    transformed_scores = pathway_scoring.compute_rank_pathway_scores(
        np.exp(expression),
        genesets,
    )

    expected = pd.DataFrame(
        {"LOW": [-0.125, 0.375], "HIGH": [0.375, -0.125]},
        index=expression.index,
    )
    pd.testing.assert_frame_equal(scores, expected)
    pd.testing.assert_frame_equal(transformed_scores, expected)


def test_robust_pathway_scaler_applies_training_statistics_to_external_scores():
    pathway_scoring = _load_module()
    training = pd.DataFrame(
        {"P1": [0.0, 1.0, 2.0], "P2": [2.0, 4.0, 6.0]},
        index=["T1", "T2", "T3"],
    )
    external = pd.DataFrame(
        {"P1": [3.0], "P2": [8.0]},
        index=["E1"],
    )

    scaler = pathway_scoring.RobustPathwayScaler.fit(training)

    pd.testing.assert_frame_equal(
        scaler.transform(training),
        pd.DataFrame(
            {"P1": [-1.0, 0.0, 1.0], "P2": [-1.0, 0.0, 1.0]},
            index=training.index,
        ),
    )
    pd.testing.assert_frame_equal(
        scaler.transform(external),
        pd.DataFrame({"P1": [2.0], "P2": [2.0]}, index=external.index),
    )
    assert scaler.to_dict() == {
        "method": "training-median-iqr-v1",
        "medians": {"P1": 1.0, "P2": 4.0},
        "iqrs": {"P1": 1.0, "P2": 2.0},
    }


def test_robust_pathway_scaler_round_trips_checkpoint_contract():
    pathway_scoring = _load_module()
    payload = {
        "method": "training-median-iqr-v1",
        "medians": {"P1": 1.0, "P2": 4.0},
        "iqrs": {"P1": 1.0, "P2": 2.0},
    }

    scaler = pathway_scoring.RobustPathwayScaler.from_dict(payload)

    assert scaler.to_dict() == payload
    with pytest.raises(ValueError, match="Unsupported pathway scaler"):
        pathway_scoring.RobustPathwayScaler.from_dict(
            {**payload, "method": "external-cohort-scaler"}
        )


def test_rank_pathway_scores_ignore_missing_genes_but_reject_infinity():
    pathway_scoring = _load_module()
    expression = pd.DataFrame(
        {
            "G1": [1.0, np.nan],
            "G2": [2.0, 2.0],
            "G3": [np.nan, 3.0],
        },
        index=["S1", "S2"],
    )

    scores = pathway_scoring.compute_rank_pathway_scores(
        expression,
        {"P12": ["G1", "G2"], "P3": ["G3"]},
    )

    pd.testing.assert_frame_equal(
        scores,
        pd.DataFrame(
            {"P12": [0.25, 0.0], "P3": [0.0, 0.5]},
            index=expression.index,
        ),
    )
    with pytest.raises(ValueError, match="infinite"):
        pathway_scoring.compute_rank_pathway_scores(
            pd.DataFrame({"G1": [np.inf]}),
            {"P": ["G1"]},
        )
