import importlib
import importlib.util
import json
import time

import pandas as pd
import pytest


def _adapter():
    spec = importlib.util.find_spec("pipeline.metabric_adapter")
    assert spec is not None, "pipeline.metabric_adapter must exist"
    return importlib.import_module("pipeline.metabric_adapter")


def _write_raw_study(raw_dir):
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "Hugo_Symbol": ["G1", "G2"],
            "Entrez_Gene_Id": [1, 2],
            "SAMPLE-A": [1.0, 5.0],
            "SAMPLE-B": [3.0, 7.0],
        }
    ).to_csv(
        raw_dir / "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        sep="\t",
        index=False,
    )
    (raw_dir / "data_clinical_patient.txt").write_text(
        "PATIENT_ID\tOS_MONTHS\tOS_STATUS\n"
        "SAMPLE-A\t10\t1:DECEASED\n"
        "SAMPLE-B\t5\t0:LIVING\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53"],
            "Tumor_Sample_Barcode": ["SAMPLE-A"],
            "HGVSp_Short": ["p.R175H"],
            "Consequence": ["missense_variant"],
        }
    ).to_csv(
        raw_dir / "data_mutations.txt",
        sep="\t",
        index=False,
    )


def test_build_pathway_scores_uses_shared_single_sample_gene_ranks(tmp_path):
    expression_path = tmp_path / "expression.txt"
    pd.DataFrame(
        {
            "Hugo_Symbol": ["G1", "G1", "G2", "OTHER"],
            "Entrez_Gene_Id": [1, 1, 2, 3],
            "SAMPLE-A": [1.0, 3.0, 5.0, 100.0],
            "SAMPLE-B": [3.0, 5.0, 1.0, 200.0],
        }
    ).to_csv(expression_path, sep="\t", index=False)
    genesets = {
        "PATHWAY-1": ["G1", "G2"],
        "PATHWAY-2": ["G2", "MISSING"],
    }

    scores = _adapter().build_metabric_pathway_scores(
        expression_path,
        genesets,
    )

    expected = pd.DataFrame(
        {
            "PATHWAY-1": [0.25, 0.25],
            "PATHWAY-2": [0.5, 0.0],
        },
        index=pd.Index(["SAMPLE-A", "SAMPLE-B"], name="sample_id"),
    )
    pd.testing.assert_frame_equal(scores, expected)


def test_build_clinical_parses_overall_survival_and_drops_invalid_rows(tmp_path):
    clinical_path = tmp_path / "clinical.txt"
    clinical_path.write_text(
        "# cBioPortal metadata\n"
        "PATIENT_ID\tOS_MONTHS\tOS_STATUS\n"
        "SAMPLE-A\t10\t1:DECEASED\n"
        "SAMPLE-B\t5\t0:LIVING\n"
        "SAMPLE-C\tNA\t1:DECEASED\n",
        encoding="utf-8",
    )

    clinical = _adapter().build_metabric_clinical(clinical_path)

    expected = pd.DataFrame(
        {
            "time": [304.4, 152.2],
            "event": [1.0, 0.0],
        },
        index=pd.Index(["SAMPLE-A", "SAMPLE-B"], name="sample_id"),
    )
    pd.testing.assert_frame_equal(clinical, expected)


def test_build_mutations_prefers_protein_change_then_consequence(tmp_path):
    mutations_path = tmp_path / "mutations.txt"
    pd.DataFrame(
        {
            "Hugo_Symbol": ["TP53", "PIK3CA", "GATA3"],
            "Tumor_Sample_Barcode": ["SAMPLE-A", "SAMPLE-A", "SAMPLE-B"],
            "HGVSp_Short": ["p.R175H", None, ""],
            "Consequence": ["missense_variant", "intron_variant", None],
        }
    ).to_csv(mutations_path, sep="\t", index=False)

    mutations = _adapter().build_metabric_mutations(mutations_path)

    assert mutations == {
        "SAMPLE-A": [
            ["TP53", "p.R175H"],
            ["PIK3CA", "intron_variant"],
        ],
        "SAMPLE-B": [["GATA3", ""]],
    }


def test_prepare_metabric_writes_and_reuses_valid_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_study(raw_dir)
    genesets_path = tmp_path / "genesets.json"
    genesets_path.write_text(
        json.dumps({"PATHWAY-1": ["G1", "G2"]}),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    adapter = _adapter()
    assert hasattr(adapter, "prepare_metabric"), (
        "adapter must prepare and cache the METABRIC cohort"
    )

    cohort = adapter.prepare_metabric(raw_dir, genesets_path, cache_dir)

    assert cohort == {
        "expr": str(cache_dir / "pathway_scores.csv"),
        "clinical": str(cache_dir / "clinical.csv"),
        "mutations": str(cache_dir / "mutations.json"),
        "precomputed_pathways": True,
    }
    assert (cache_dir / "manifest.json").is_file()
    first_mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in cache_dir.iterdir()
    }

    time.sleep(0.01)
    repeated = adapter.prepare_metabric(raw_dir, genesets_path, cache_dir)

    assert repeated == cohort
    assert {
        path.name: path.stat().st_mtime_ns
        for path in cache_dir.iterdir()
    } == first_mtimes


def test_prepare_metabric_rebuilds_when_pathway_genesets_change(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_study(raw_dir)
    genesets_path = tmp_path / "genesets.json"
    genesets_path.write_text(
        json.dumps({"PATHWAY-1": ["G1"]}),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    adapter = _adapter()
    assert hasattr(adapter, "prepare_metabric"), (
        "adapter must invalidate stale METABRIC caches"
    )
    adapter.prepare_metabric(raw_dir, genesets_path, cache_dir)

    genesets_path.write_text(
        json.dumps({"PATHWAY-1": ["G1"], "PATHWAY-2": ["G2"]}),
        encoding="utf-8",
    )
    adapter.prepare_metabric(raw_dir, genesets_path, cache_dir)

    rebuilt_scores = pd.read_csv(
        cache_dir / "pathway_scores.csv",
        index_col=0,
    )
    assert rebuilt_scores.columns.tolist() == ["PATHWAY-1", "PATHWAY-2"]
