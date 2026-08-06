import gzip
import importlib
import importlib.util
import json
import time

import numpy as np
import pandas as pd


CLINICAL_COLUMNS = [
    "PID",
    "GEO asscession number",
    "lymph node status",
    "time to relapse or last follow-up (months)",
    "relapse (1=True)",
    "ER Status",
    "Brain relapses (1=yes, 0=no)",
]


def _adapter():
    spec = importlib.util.find_spec("pipeline.geo_gse2034_adapter")
    assert spec is not None, "pipeline.geo_gse2034_adapter must exist"
    return importlib.import_module("pipeline.geo_gse2034_adapter")


def _write_soft_file(path):
    lines = [
        "^SERIES = GSE2034",
        "!series_table_begin = Patient clinical parameters header descriptions",
        "\t".join(CLINICAL_COLUMNS),
        "1\tGSM1\tnegative\t10\t0\tER+\t0",
        "2\tGSM2\tnegative\t5\t1\tER-\t0",
        "!series_table_end",
        "^PLATFORM = GPL96",
        "!platform_table_begin",
        "ID\tGene Symbol\tGene Title",
        "P1\tG1\tGene one",
        "P2\tG1 /// G2\tAmbiguous probe",
        "P3\tG2\tGene two",
        "P4\tOTHER\tOther gene",
        "!platform_table_end",
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_raw_study(raw_dir):
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "GSM1": [1.0, 4.0, 16.0, 64.0],
            "GSM2": [4.0, 16.0, 4.0, 64.0],
        },
        index=pd.Index(["P1", "P2", "P3", "P4"], name="ID_REF"),
    ).to_csv(raw_dir / "GSE2034_expression_matrix.csv")
    _write_soft_file(raw_dir / "GSE2034_family.soft.gz")


def test_load_soft_tables_parses_rfs_and_multigene_probe_annotations(tmp_path):
    soft_path = tmp_path / "study.soft.gz"
    _write_soft_file(soft_path)

    clinical_table, probe_to_genes = _adapter().load_gse2034_soft_tables(
        soft_path
    )

    assert clinical_table["GEO asscession number"].tolist() == ["GSM1", "GSM2"]
    assert clinical_table["relapse (1=True)"].tolist() == ["0", "1"]
    assert probe_to_genes == {
        "P1": ["G1"],
        "P2": ["G1", "G2"],
        "P3": ["G2"],
        "P4": ["OTHER"],
    }


def test_build_pathway_scores_aggregates_genes_then_uses_shared_ranks(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_study(raw_dir)
    _, probe_to_genes = _adapter().load_gse2034_soft_tables(
        raw_dir / "GSE2034_family.soft.gz"
    )
    genesets = {
        "PATHWAY-1": ["G1", "G2"],
        "PATHWAY-2": ["G2"],
    }

    scores = _adapter().build_geo_gse2034_pathway_scores(
        raw_dir / "GSE2034_expression_matrix.csv",
        probe_to_genes,
        genesets,
    )

    expected = pd.DataFrame(
        {
            "PATHWAY-1": [0.25, 0.25],
            "PATHWAY-2": [0.5, 0.25],
        },
        index=pd.Index(["GSM1", "GSM2"], name="sample_id"),
    )
    pd.testing.assert_frame_equal(scores, expected)


def test_build_clinical_uses_relapse_free_survival_months(tmp_path):
    soft_path = tmp_path / "study.soft.gz"
    _write_soft_file(soft_path)
    clinical_table, _ = _adapter().load_gse2034_soft_tables(soft_path)

    clinical = _adapter().build_geo_gse2034_clinical(clinical_table)

    expected = pd.DataFrame(
        {"time": [10.0, 5.0], "event": [0.0, 1.0]},
        index=pd.Index(["GSM1", "GSM2"], name="sample_id"),
    )
    pd.testing.assert_frame_equal(clinical, expected)


def test_prepare_geo_gse2034_aligns_inputs_and_reuses_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_study(raw_dir)
    genesets_path = tmp_path / "genesets.json"
    genesets_path.write_text(
        json.dumps({"PATHWAY-1": ["G1", "G2"]}),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"

    cohort = _adapter().prepare_geo_gse2034(
        raw_dir,
        genesets_path,
        cache_dir,
    )

    assert cohort == {
        "expr": str(cache_dir / "pathway_scores.csv"),
        "clinical": str(cache_dir / "clinical.csv"),
        "mutations": None,
        "precomputed_pathways": True,
    }
    scores = pd.read_csv(cohort["expr"], index_col=0)
    clinical = pd.read_csv(cohort["clinical"], index_col=0)
    assert scores.index.tolist() == ["GSM1", "GSM2"]
    assert clinical.index.tolist() == ["GSM1", "GSM2"]
    assert (cache_dir / "manifest.json").is_file()
    first_mtimes = {
        path.name: path.stat().st_mtime_ns for path in cache_dir.iterdir()
    }

    time.sleep(0.01)
    repeated = _adapter().prepare_geo_gse2034(
        raw_dir,
        genesets_path,
        cache_dir,
    )

    assert repeated == cohort
    assert {
        path.name: path.stat().st_mtime_ns for path in cache_dir.iterdir()
    } == first_mtimes
