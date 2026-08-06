import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace

from pipeline import evaluate
from src.pathway_scoring import RobustPathwayScaler


def test_validate_checkpoint_tcga_split_rejects_legacy_model():
    manifest = {
        "version": 1,
        "seed": 42,
        "n_total": 868,
        "n_train": 694,
        "n_test": 174,
        "total_events": 102,
        "train_events": 82,
        "test_events": 20,
    }

    assert hasattr(evaluate, "validate_checkpoint_tcga_split"), (
        "evaluation must reject a model that was not trained on the fixed split"
    )

    with pytest.raises(ValueError, match="retrain the model"):
        evaluate.validate_checkpoint_tcga_split(
            {"model_state_dict": {}},
            manifest,
        )


def test_validate_checkpoint_tcga_split_accepts_matching_model():
    split_metadata = {
        "version": 1,
        "seed": 42,
        "n_total": 868,
        "n_train": 694,
        "n_test": 174,
        "total_events": 102,
        "train_events": 82,
        "test_events": 20,
    }

    assert hasattr(evaluate, "validate_checkpoint_tcga_split"), (
        "evaluation must verify the model's TCGA split metadata"
    )

    evaluate.validate_checkpoint_tcga_split(
        {"model_state_dict": {}, "tcga_split": dict(split_metadata)},
        dict(split_metadata),
    )


def test_load_evaluation_data_uses_shared_exact_tcga_split(tmp_path):
    patient_ids = [f"TCGA-{index:02d}" for index in range(10)]
    expression = pd.DataFrame(
        {
            "GENE_A": list(range(10)),
            "GENE_B": list(range(10, 20)),
        },
        index=pd.Index(patient_ids, name="case_id"),
    )
    clinical = pd.DataFrame(
        {
            "case_id": patient_ids,
            "survival_months": list(range(1, 11)),
            "censorship": [0] * 4 + [1] * 6,
            # Deliberately incompatible with the requested 8/2 split.
            "train": [1.0] * 9 + [0.0],
        }
    )
    expression_path = tmp_path / "expression.csv"
    clinical_path = tmp_path / "clinical.csv"
    expression.to_csv(expression_path)
    clinical.to_csv(clinical_path, index=False)

    assert hasattr(evaluate, "load_evaluation_data"), (
        "evaluation needs a loader that applies the TCGA holdout split"
    )

    split_options = {
        "tcga_split_manifest_path": str(tmp_path / "tcga_split.json"),
        "tcga_n_train": 8,
        "tcga_n_test": 2,
        "tcga_split_seed": 42,
    }
    train_expression, train_clinical = evaluate.load_evaluation_data(
        str(expression_path),
        str(clinical_path),
        clinical_split="train",
        **split_options,
    )
    test_expression, test_clinical = evaluate.load_evaluation_data(
        str(expression_path),
        str(clinical_path),
        clinical_split="test",
        **split_options,
    )

    train_ids = set(train_expression.index)
    test_ids = set(test_expression.index)
    assert len(train_expression) == len(train_clinical) == 8
    assert len(test_expression) == len(test_clinical) == 2
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(patient_ids)
    assert int(train_clinical["event"].sum()) == 3
    assert int(test_clinical["event"].sum()) == 1


def test_build_evaluation_dataset_uses_precomputed_pathway_scores():
    expression = pd.DataFrame(
        {
            "PATHWAY-1": [1.5, 2.5],
            "PATHWAY-2": [3.5, 4.5],
        },
        index=["SAMPLE-A", "SAMPLE-B"],
    )
    clinical = pd.DataFrame(
        {
            "time": [100.0, 200.0],
            "event": [1.0, 0.0],
        },
        index=["SAMPLE-A", "SAMPLE-B"],
    )
    train_meta = SimpleNamespace(
        gene_means=pd.Series({"GENE-A": 0.0}),
        gene_stds=pd.Series({"GENE-A": 1.0}),
    )
    assert hasattr(evaluate, "build_evaluation_dataset"), (
        "evaluation must explicitly support precomputed pathway matrices"
    )

    dataset = evaluate.build_evaluation_dataset(
        expression=expression,
        clinical=clinical,
        mutations={},
        genesets={"PATHWAY-1": ["GENE-A"], "PATHWAY-2": ["GENE-B"]},
        train_ds_meta=train_meta,
        precomputed_pathways=True,
    )

    assert dataset.precomputed_pathways is True
    pd.testing.assert_frame_equal(dataset.pathway_scores_df, expression)


def test_build_evaluation_dataset_restores_training_rank_scaler():
    expression = pd.DataFrame(
        {"P1": [2.0], "P2": [8.0]},
        index=["EXTERNAL-1"],
    )
    clinical = pd.DataFrame(
        {"time": [100.0], "event": [1.0]},
        index=expression.index,
    )
    scaler = RobustPathwayScaler(
        medians=pd.Series({"P1": 1.0, "P2": 4.0}),
        iqrs=pd.Series({"P1": 1.0, "P2": 2.0}),
    )
    train_meta = SimpleNamespace(
        gene_means=None,
        gene_stds=None,
        pathway_scaler=scaler,
    )

    dataset = evaluate.build_evaluation_dataset(
        expression=expression,
        clinical=clinical,
        mutations={},
        genesets={"P1": ["G1"], "P2": ["G2"]},
        train_ds_meta=train_meta,
        precomputed_pathways=True,
        pathway_scoring_method="rank",
    )

    pd.testing.assert_frame_equal(
        dataset.pathway_scores_df,
        pd.DataFrame({"P1": [1.0], "P2": [2.0]}, index=expression.index),
    )


def test_bootstrap_cindex_is_deterministic_and_uses_risk_direction():
    times = np.arange(8.0, 0.0, -1.0)
    events = np.array([1, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    risk = np.arange(1.0, 9.0)

    first = evaluate.bootstrap_cindex(
        times, events, risk, n_bootstrap=100, seed=42
    )
    second = evaluate.bootstrap_cindex(
        times, events, risk, n_bootstrap=100, seed=42
    )

    assert first == second
    assert first["c_index"] == pytest.approx(1.0)
    assert first["ci_lower"] == pytest.approx(1.0)
    assert first["ci_upper"] == pytest.approx(1.0)
    assert first["n_valid_bootstrap"] > 0
