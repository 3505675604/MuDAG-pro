import numpy as np
import pandas as pd
import pytest
import torch

from pipeline import train_tcga
from src.cox_model import RidgeCoxSurvivalModel
from src.dataset import PathwayDataset
from src.graph_propagation import DirectedPathwayPropagation
from src.personalized_graph import PersonalizedGraphModulator


def test_load_tcga_clinical_data_normalizes_and_filters_training_cases(tmp_path):
    raw = pd.DataFrame(
        {
            "case_id": ["TCGA-A", "TCGA-A", "TCGA-B", "TCGA-C"],
            "slide_id": ["A-1", "A-2", "B-1", "C-1"],
            "survival_months": [10.0, 10.0, 20.0, 30.0],
            "censorship": [0, 0, 1, 0],
            "train": [1.0, 1.0, 1.0, 0.0],
        }
    )
    clinical_path = tmp_path / "tcga_brca.csv"
    raw.to_csv(clinical_path, index=False)

    assert hasattr(train_tcga, "load_tcga_clinical_data"), (
        "training needs a loader that converts raw TCGA clinical metadata"
    )

    clinical = train_tcga.load_tcga_clinical_data(str(clinical_path))

    assert clinical.index.tolist() == ["TCGA-A", "TCGA-B"]
    assert clinical.index.name == "case_id"
    assert clinical["event"].tolist() == [1.0, 0.0]
    assert clinical["time"].tolist() == pytest.approx([304.4, 608.8])


def test_load_tcga_clinical_data_accepts_standardized_schema(tmp_path):
    standardized = pd.DataFrame(
        {
            "time": [100.0, 200.0],
            "event": [1.0, 0.0],
        },
        index=pd.Index(["TCGA-A", "TCGA-B"], name="case_id"),
    )
    clinical_path = tmp_path / "clinical.csv"
    standardized.to_csv(clinical_path)

    assert hasattr(train_tcga, "load_tcga_clinical_data"), (
        "training needs a loader that accepts normalized clinical metadata"
    )

    clinical = train_tcga.load_tcga_clinical_data(str(clinical_path))

    pd.testing.assert_frame_equal(clinical, standardized)


def test_load_patient_expression_data_averages_duplicate_patients(tmp_path):
    raw = pd.DataFrame(
        {
            "GENE_A": [1.0, 3.0, 10.0],
            "GENE_B": [2.0, 6.0, 20.0],
        },
        index=pd.Index(["TCGA-A", "TCGA-A", "TCGA-B"], name="case_id"),
    )
    expression_path = tmp_path / "expression.csv"
    raw.to_csv(expression_path)

    assert hasattr(train_tcga, "load_patient_expression_data"), (
        "training needs a loader that collapses duplicate patient profiles"
    )

    expression = train_tcga.load_patient_expression_data(str(expression_path))

    expected = pd.DataFrame(
        {
            "GENE_A": [2.0, 10.0],
            "GENE_B": [4.0, 20.0],
        },
        index=pd.Index(["TCGA-A", "TCGA-B"], name="case_id"),
    )
    pd.testing.assert_frame_equal(expression, expected)


def test_prepare_tcga_split_aligns_before_exact_stratified_split(tmp_path):
    patient_ids = [f"TCGA-{index:02d}" for index in range(12)]
    clinical = pd.DataFrame(
        {
            "time": np.arange(1, 13, dtype=float),
            "event": [1.0] * 4 + [0.0] * 8,
            # This legacy column deliberately describes the wrong split.
            "train": [1.0] * 9 + [0.0] * 3,
        },
        index=patient_ids,
    )
    expression = pd.DataFrame(
        {"GENE_A": np.arange(10, dtype=float)},
        index=patient_ids[:10],
    )
    manifest_path = tmp_path / "tcga_split.json"

    assert hasattr(train_tcga, "prepare_tcga_split"), (
        "TCGA splitting must happen after expression/clinical alignment"
    )

    split = train_tcga.prepare_tcga_split(
        expression,
        clinical,
        manifest_path=str(manifest_path),
        n_train=8,
        n_test=2,
        seed=42,
    )

    train_ids = set(split.train_expression.index)
    test_ids = set(split.test_expression.index)
    assert len(train_ids) == 8
    assert len(test_ids) == 2
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(patient_ids[:10])
    assert int(split.train_clinical["event"].sum()) == 3
    assert int(split.test_clinical["event"].sum()) == 1
    assert manifest_path.exists()

    # Reordering source rows must not change the persisted patient assignment.
    reused = train_tcga.prepare_tcga_split(
        expression.sample(frac=1.0, random_state=7),
        clinical.sample(frac=1.0, random_state=9),
        manifest_path=str(manifest_path),
        n_train=8,
        n_test=2,
        seed=42,
    )
    assert reused.train_expression.index.tolist() == split.train_expression.index.tolist()
    assert reused.test_expression.index.tolist() == split.test_expression.index.tolist()


def test_prepare_tcga_split_rejects_wrong_aligned_total(tmp_path):
    expression = pd.DataFrame(
        {"GENE_A": [1.0, 2.0, 3.0]},
        index=["TCGA-A", "TCGA-B", "TCGA-C"],
    )
    clinical = pd.DataFrame(
        {"time": [1.0, 2.0, 3.0], "event": [1.0, 0.0, 0.0]},
        index=expression.index,
    )

    assert hasattr(train_tcga, "prepare_tcga_split"), (
        "TCGA split validation must reject a cohort other than n_train + n_test"
    )

    with pytest.raises(ValueError, match="aligned TCGA cohort has 3 patients"):
        train_tcga.prepare_tcga_split(
            expression,
            clinical,
            manifest_path=str(tmp_path / "tcga_split.json"),
            n_train=2,
            n_test=2,
            seed=42,
        )


def test_train_epoch_uses_one_complete_cox_risk_set_across_feature_chunks():
    expression = pd.DataFrame(
        {
            "PATHWAY-A": [2.0, 1.0, 0.0],
            "PATHWAY-B": [0.0, 1.0, 2.0],
        },
        index=["S1", "S2", "S3"],
    )
    clinical = pd.DataFrame(
        {"time": [1.0, 2.0, 3.0], "event": [1.0, 1.0, 1.0]},
        index=expression.index,
    )
    genesets = {"PATHWAY-A": [], "PATHWAY-B": []}
    dataset = PathwayDataset(
        expression,
        genesets,
        clinical,
        mutation_dict={},
        precomputed_pathways=True,
    )
    model = RidgeCoxSurvivalModel(in_features=2, l2_reg=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_tcga.train_epoch(
        model=model,
        propagator=DirectedPathwayPropagation(alpha=0.0),
        modulator=PersonalizedGraphModulator(
            list(genesets), [], {}
        ),
        A_enhanced=np.zeros((2, 2), dtype=np.float32),
        dataset=dataset,
        batch_size=1,
        optimizer=optimizer,
    )

    assert torch.count_nonzero(model.linear.weight).item() == 2


def test_build_cv_splits_stratifies_each_validation_fold_by_event():
    events = np.array([1, 0, 0, 0, 0] * 5)

    splits = train_tcga.build_cv_splits(events, n_folds=5, seed=42)

    assert len(splits) == 5
    assert [int(events[val_idx].sum()) for _, val_idx in splits] == [1] * 5
    assert [len(val_idx) for _, val_idx in splits] == [5] * 5


def test_build_propagated_features_preserves_order_across_memory_chunks():
    expression = pd.DataFrame(
        {"PATHWAY-A": [2.0, 4.0], "PATHWAY-B": [3.0, 5.0]},
        index=["S1", "S2"],
    )
    clinical = pd.DataFrame(
        {"time": [1.0, 2.0], "event": [1.0, 0.0]},
        index=expression.index,
    )
    genesets = {"PATHWAY-A": [], "PATHWAY-B": []}
    dataset = PathwayDataset(
        expression,
        genesets,
        clinical,
        mutation_dict={},
        precomputed_pathways=True,
    )

    features = train_tcga.build_propagated_features(
        propagator=DirectedPathwayPropagation(alpha=0.5),
        modulator=PersonalizedGraphModulator(list(genesets), [], {}),
        A_enhanced=np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        dataset=dataset,
        batch_size=1,
        device="cpu",
    )

    torch.testing.assert_close(
        features,
        torch.tensor([[2.0, 4.0], [4.0, 7.0]]),
    )


def test_summarize_personalization_reports_profiles_rules_and_changed_edges():
    rules = [
        {
            "source": "PATHWAY-A",
            "target": "PATHWAY-B",
            "gene": "PIK3CA",
            "delta": 1.0,
            "confidence": 1.0,
        }
    ]
    modulator = PersonalizedGraphModulator(
        ["PATHWAY-A", "PATHWAY-B"],
        rules,
        {"PIK3CA": ["PATHWAY-A"]},
    )
    mutations = {
        "S1": [("PIK3CA", "missense_variant")],
        "S2": [("TP53", "stop_gained")],
    }

    summary = train_tcga.summarize_personalization(
        sample_ids=["S1", "S2", "S3"],
        mutations=mutations,
        modulator=modulator,
        A_enhanced=np.array(
            [[0.0, 1.0], [0.0, 0.0]], dtype=np.float32
        ),
    )

    assert summary == {
        "samples": 3,
        "mutation_profiles_matched": 2,
        "patients_with_rule_matches": 1,
        "patients_with_modulated_edges": 1,
        "active_rule_edges": 1,
    }


def test_select_final_training_epochs_uses_median_best_fold_early_stop():
    cv_results = [
        {
            "alpha": 0.0,
            "l2_reg": 1.0,
            "fold_epochs": [51, 54, 54, 113, 92],
        },
        {
            "alpha": 0.1,
            "l2_reg": 1.0,
            "fold_epochs": [200, 200, 200, 200, 200],
        },
    ]

    epochs = train_tcga.select_final_training_epochs(
        cv_results,
        best_hyperparams={"alpha": 0.0, "l2_reg": 1.0},
        max_epochs=500,
    )

    assert epochs == 54


def test_fit_graph_elastic_net_ensemble_uses_rank_contract(monkeypatch):
    sample_ids = [f"S{i}" for i in range(6)]
    expression = pd.DataFrame(
        {
            "G1": [1, 2, 3, 4, 5, 6],
            "G2": [6, 5, 4, 3, 2, 1],
            "G3": [2, 1, 4, 3, 6, 5],
        },
        index=sample_ids,
        dtype=float,
    )
    clinical = pd.DataFrame(
        {
            "time": [60, 50, 40, 30, 20, 10],
            "event": [1, 0, 1, 1, 0, 1],
        },
        index=sample_ids,
        dtype=float,
    )
    observed = {}

    def fake_fit(features, times, events, **kwargs):
        observed["features"] = features.copy()
        observed["kwargs"] = kwargs
        return {
            "model_type": "elastic_net_cox_ensemble_v1",
            "pathway_names": list(features.columns),
            "best_penalizer": 0.01,
            "best_l1_ratio": 0.5,
            "best_cv_c_index": 0.65,
            "stable_pathways": [features.columns[0]],
            "members": [{
                "coefficients": [1.0, 0.0],
                "training_risk_mean": 0.0,
                "training_risk_std": 1.0,
            }],
        }

    monkeypatch.setattr(
        train_tcga,
        "fit_elastic_net_cox_ensemble",
        fake_fit,
        raising=False,
    )

    checkpoint = train_tcga.fit_graph_elastic_net_ensemble(
        expression=expression,
        clinical=clinical,
        genesets={"P1": ["G1", "G2"], "P2": ["G3"]},
        mutations={},
        modulator=object(),
        A_enhanced=np.zeros((2, 2), dtype=np.float32),
        alpha_grid=[0.0],
        penalizer_grid=[0.01],
        l1_ratio_grid=[0.5],
        n_folds=3,
        cv_repeats=2,
        min_selection_frequency=0.5,
        seed=42,
        batch_size=4,
        device="cpu",
    )

    assert checkpoint["model_type"] == "elastic_net_cox_ensemble_v1"
    assert checkpoint["alpha"] == 0.0
    assert checkpoint["pathway_scaler"]["method"] == "training-median-iqr-v1"
    assert np.isfinite(checkpoint["train_c_index"])
    assert observed["features"].columns.tolist() == ["P1", "P2"]
    assert observed["kwargs"] == {
        "penalizer_grid": [0.01],
        "l1_ratio_grid": [0.5],
        "n_splits": 3,
        "n_repeats": 2,
        "seed": 42,
        "min_selection_frequency": 0.5,
    }
