import importlib
import importlib.util
import json
import time

import pandas as pd


TITLE_COLUMN = "title"
TIME_COLUMN = "characteristics_ch1.21.overall survival days"
EVENT_COLUMN = "characteristics_ch1.22.overall survival event"


def _adapter():
    spec = importlib.util.find_spec("pipeline.scan_b_adapter")
    assert spec is not None, "pipeline.scan_b_adapter must exist"
    return importlib.import_module("pipeline.scan_b_adapter")


def _write_raw_study(raw_dir):
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "F1": [1.0, 5.0, 100.0],
            "F2": [3.0, 1.0, 200.0],
            "F1repl": [9.0, 9.0, 300.0],
        },
        index=["G1", "G2", "OTHER"],
    ).to_csv(
        raw_dir
        / "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz",
        compression="gzip",
    )
    pd.DataFrame(
        {
            TITLE_COLUMN: ["F1", "F2", "F1repl"],
            TIME_COLUMN: [100, 200, 100],
            EVENT_COLUMN: [1, 0, 1],
        },
        index=["GSM1", "GSM2", "GSM3"],
    ).to_csv(raw_dir / "GSE96058_clinical_metadata.csv")


def test_build_pathway_scores_uses_shared_ranks_and_excludes_replicates(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_study(raw_dir)
    genesets = {
        "PATHWAY-1": ["G1", "G2"],
        "PATHWAY-2": ["G2", "MISSING"],
    }

    scores = _adapter().build_scan_b_pathway_scores(
        raw_dir
        / "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz",
        genesets,
        chunk_size=1,
    )

    expected = pd.DataFrame(
        {
            "PATHWAY-1": [0.25, 0.25],
            "PATHWAY-2": [0.5, 0.0],
        },
        index=pd.Index(["F1", "F2"], name="sample_id"),
    )
    pd.testing.assert_frame_equal(scores, expected)


def test_build_clinical_uses_overall_survival_and_excludes_replicates(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_study(raw_dir)

    clinical = _adapter().build_scan_b_clinical(
        raw_dir / "GSE96058_clinical_metadata.csv"
    )

    expected = pd.DataFrame(
        {
            "time": [100.0, 200.0],
            "event": [1.0, 0.0],
        },
        index=pd.Index(["F1", "F2"], name="sample_id"),
    )
    pd.testing.assert_frame_equal(clinical, expected)


def test_prepare_scan_b_aligns_inputs_and_reuses_valid_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_study(raw_dir)
    genesets_path = tmp_path / "genesets.json"
    genesets_path.write_text(
        json.dumps({"PATHWAY-1": ["G1", "G2"]}),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"

    cohort = _adapter().prepare_scan_b(raw_dir, genesets_path, cache_dir)

    assert cohort == {
        "expr": str(cache_dir / "pathway_scores.csv"),
        "clinical": str(cache_dir / "clinical.csv"),
        "mutations": None,
        "precomputed_pathways": True,
    }
    scores = pd.read_csv(cohort["expr"], index_col=0)
    clinical = pd.read_csv(cohort["clinical"], index_col=0)
    assert scores.index.tolist() == ["F1", "F2"]
    assert clinical.index.tolist() == ["F1", "F2"]
    assert (cache_dir / "manifest.json").is_file()
    first_mtimes = {
        path.name: path.stat().st_mtime_ns for path in cache_dir.iterdir()
    }

    time.sleep(0.01)
    repeated = _adapter().prepare_scan_b(raw_dir, genesets_path, cache_dir)

    assert repeated == cohort
    assert {
        path.name: path.stat().st_mtime_ns for path in cache_dir.iterdir()
    } == first_mtimes
