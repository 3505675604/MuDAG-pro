import importlib

import pandas as pd
import pytest


def _load_module():
    try:
        return importlib.import_module("pipeline.cohort_manifest")
    except ModuleNotFoundError:
        pytest.fail("pipeline.cohort_manifest must implement frozen cohort selection")


def _candidate_frames():
    patient_ids = [f"S{index}" for index in range(1, 9)]
    expression = pd.DataFrame(
        {"PATHWAY": list(range(8))},
        index=patient_ids,
    )
    clinical = pd.DataFrame(
        {
            "time": list(range(10, 90, 10)),
            "event": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=patient_ids,
    )
    return expression, clinical


def test_freeze_external_cohort_selects_exact_strata_independent_of_row_order(
    tmp_path,
):
    cohort_manifest = _load_module()
    expression, clinical = _candidate_frames()
    manifest_path = tmp_path / "cohort.json"

    selected = cohort_manifest.freeze_external_cohort(
        expression.sample(frac=1.0, random_state=3),
        clinical.sample(frac=1.0, random_state=5),
        manifest_path=str(manifest_path),
        dataset="TEST",
        n_events=2,
        n_censored=3,
    )

    assert selected.expression.index.tolist() == ["S1", "S3", "S5", "S6", "S8"]
    assert selected.clinical.index.tolist() == ["S1", "S3", "S5", "S6", "S8"]
    assert int(selected.clinical["event"].sum()) == 2
    assert selected.manifest["n_samples"] == 5
    assert selected.manifest["n_events"] == 2
    assert selected.manifest["n_censored"] == 3

    reused = cohort_manifest.freeze_external_cohort(
        expression,
        clinical,
        manifest_path=str(manifest_path),
        dataset="TEST",
        n_events=2,
        n_censored=3,
    )
    assert reused.manifest == selected.manifest


def test_freeze_external_cohort_rejects_candidate_id_drift(tmp_path):
    cohort_manifest = _load_module()
    expression, clinical = _candidate_frames()
    manifest_path = tmp_path / "cohort.json"
    cohort_manifest.freeze_external_cohort(
        expression,
        clinical,
        manifest_path=str(manifest_path),
        dataset="TEST",
        n_events=2,
        n_censored=3,
    )

    changed_expression = expression.rename(index={"S8": "S9"})
    changed_clinical = clinical.rename(index={"S8": "S9"})
    with pytest.raises(ValueError, match="candidate cohort has changed"):
        cohort_manifest.freeze_external_cohort(
            changed_expression,
            changed_clinical,
            manifest_path=str(manifest_path),
            dataset="TEST",
            n_events=2,
            n_censored=3,
        )
