"""Deterministic, auditable patient selection for fixed external cohorts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class FrozenCohort:
    expression: pd.DataFrame
    clinical: pd.DataFrame
    manifest: Dict


def _patient_hash(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()


def _candidate_signature(clinical: pd.DataFrame) -> str:
    records = [
        [patient_id, float(row.time), int(row.event)]
        for patient_id, row in clinical.loc[:, ["time", "event"]].iterrows()
    ]
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_external_cohort(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    manifest_path: str,
    dataset: str,
    n_events: int,
    n_censored: int,
) -> FrozenCohort:
    """Select exact event/censoring strata by stable patient-ID hash."""
    if n_events <= 0 or n_censored <= 0:
        raise ValueError("External cohort event and censored counts must be positive.")
    if not manifest_path:
        raise ValueError("An external cohort manifest path is required.")
    if not {"time", "event"}.issubset(clinical.columns):
        raise ValueError("External clinical data must contain time and event columns.")

    expression = expression.copy()
    clinical = clinical.copy()
    expression.index = expression.index.map(str)
    clinical.index = clinical.index.map(str)
    if expression.index.has_duplicates or clinical.index.has_duplicates:
        raise ValueError("External cohort patient IDs must be unique.")

    common_ids = sorted(set(expression.index) & set(clinical.index))
    if not common_ids:
        raise ValueError("External expression and clinical data have no overlapping IDs.")
    aligned_expression = expression.loc[common_ids]
    aligned_clinical = clinical.loc[common_ids]
    events = aligned_clinical["event"].astype(int)
    if not events.isin([0, 1]).all():
        raise ValueError("External cohort event labels must be binary.")
    if int(events.sum()) < n_events:
        raise ValueError(f"{dataset} has fewer than {n_events} eligible events.")
    if int((events == 0).sum()) < n_censored:
        raise ValueError(f"{dataset} has fewer than {n_censored} eligible censored cases.")

    signature = _candidate_signature(aligned_clinical)
    expected_metadata = {
        "version": 1,
        "dataset": dataset,
        "selection_method": "event-stratified-sha256-v1",
        "candidate_signature": signature,
        "n_samples": n_events + n_censored,
        "n_events": n_events,
        "n_censored": n_censored,
    }

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("candidate_signature") != signature:
            raise ValueError(f"{dataset} candidate cohort has changed since freezing.")
        for key, expected_value in expected_metadata.items():
            if manifest.get(key) != expected_value:
                raise ValueError(
                    f"{dataset} cohort manifest has invalid {key}: "
                    f"{manifest.get(key)!r}; expected {expected_value!r}."
                )
    else:
        event_ids = sorted(
            aligned_clinical.index[events == 1], key=_patient_hash
        )[:n_events]
        censored_ids = sorted(
            aligned_clinical.index[events == 0], key=_patient_hash
        )[:n_censored]
        selected_ids = sorted([*event_ids, *censored_ids])
        manifest = {**expected_metadata, "selected_ids": selected_ids}
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        temporary_path = f"{manifest_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_path, manifest_path)

    selected_ids = [str(value) for value in manifest.get("selected_ids", [])]
    if len(selected_ids) != n_events + n_censored:
        raise ValueError(f"{dataset} cohort manifest has an invalid patient count.")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"{dataset} cohort manifest contains duplicate patient IDs.")
    if not set(selected_ids).issubset(common_ids):
        raise ValueError(f"{dataset} cohort manifest contains ineligible patient IDs.")
    selected_clinical = aligned_clinical.loc[selected_ids]
    if int(selected_clinical["event"].sum()) != n_events:
        raise ValueError(f"{dataset} cohort manifest has an invalid event count.")

    return FrozenCohort(
        expression=aligned_expression.loc[selected_ids],
        clinical=selected_clinical,
        manifest=manifest,
    )
