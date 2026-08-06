from pathlib import Path

import main


def test_resolve_metabric_cohort_uses_raw_adapter_and_indexed_cache(
    tmp_path,
    monkeypatch,
):
    genesets_path = tmp_path / "genesets.json"
    genesets_path.write_text("{}", encoding="utf-8")
    prepared = {
        "expr": str(tmp_path / "pathway_scores.csv"),
        "clinical": str(tmp_path / "clinical.csv"),
        "mutations": str(tmp_path / "mutations.json"),
        "precomputed_pathways": True,
    }
    observed = {}

    def fake_prepare(raw_dir, genesets_path, cache_dir):
        observed["raw_dir"] = Path(raw_dir)
        observed["genesets_path"] = Path(genesets_path)
        observed["cache_dir"] = Path(cache_dir)
        return prepared

    monkeypatch.setattr(main, "prepare_metabric", fake_prepare, raising=False)
    assert hasattr(main, "resolve_custom_evaluation_cohorts"), (
        "main must resolve METABRIC through the raw-data adapter"
    )

    cohorts = main.resolve_custom_evaluation_cohorts(
        ["metabric"],
        base_dir=str(tmp_path),
        genesets_path=str(genesets_path),
    )

    assert cohorts == [{
        "name": "METABRIC",
        **prepared,
        "fixed_cohort": {
            "manifest_path": str(
                tmp_path
                / "data/processed/splits/metabric_1413_820_sha256.json"
            ),
            "n_events": 820,
            "n_censored": 593,
        },
    }]
    assert observed == {
        "raw_dir": tmp_path / "Data/METABRIC/metabric_raw/brca_metabric",
        "genesets_path": genesets_path,
        "cache_dir": tmp_path / "data/processed/external/metabric",
    }


def test_resolve_scan_b_cohort_uses_raw_adapter_and_static_graph(
    tmp_path,
    monkeypatch,
):
    genesets_path = tmp_path / "genesets.json"
    genesets_path.write_text("{}", encoding="utf-8")
    prepared = {
        "expr": str(tmp_path / "pathway_scores.csv"),
        "clinical": str(tmp_path / "clinical.csv"),
        "mutations": None,
        "precomputed_pathways": True,
    }
    observed = {}

    def fake_prepare(raw_dir, genesets_path, cache_dir):
        observed["raw_dir"] = Path(raw_dir)
        observed["genesets_path"] = Path(genesets_path)
        observed["cache_dir"] = Path(cache_dir)
        return prepared

    monkeypatch.setattr(main, "prepare_scan_b", fake_prepare, raising=False)

    cohorts = main.resolve_custom_evaluation_cohorts(
        ["scan_b"],
        base_dir=str(tmp_path),
        genesets_path=str(genesets_path),
    )

    assert cohorts == [{
        "name": "SCAN-B",
        **prepared,
        "fixed_cohort": {
            "manifest_path": str(
                tmp_path
                / "data/processed/splits/scan_b_2449_219_sha256.json"
            ),
            "n_events": 219,
            "n_censored": 2230,
        },
    }]
    assert observed == {
        "raw_dir": tmp_path / "Data/SCAN-B",
        "genesets_path": genesets_path,
        "cache_dir": tmp_path / "data/processed/external/scan_b",
    }


def test_resolve_geo_gse2034_cohort_uses_probe_adapter_and_static_graph(
    tmp_path,
    monkeypatch,
):
    genesets_path = tmp_path / "genesets.json"
    genesets_path.write_text("{}", encoding="utf-8")
    prepared = {
        "expr": str(tmp_path / "pathway_scores.csv"),
        "clinical": str(tmp_path / "clinical.csv"),
        "mutations": None,
        "precomputed_pathways": True,
    }
    observed = {}

    def fake_prepare(raw_dir, genesets_path, cache_dir):
        observed["raw_dir"] = Path(raw_dir)
        observed["genesets_path"] = Path(genesets_path)
        observed["cache_dir"] = Path(cache_dir)
        return prepared

    monkeypatch.setattr(
        main,
        "prepare_geo_gse2034",
        fake_prepare,
        raising=False,
    )

    cohorts = main.resolve_custom_evaluation_cohorts(
        ["geo_gse2034"],
        base_dir=str(tmp_path),
        genesets_path=str(genesets_path),
    )

    assert cohorts == [{"name": "GEO-GSE2034", **prepared}]
    assert observed == {
        "raw_dir": tmp_path / "Data/GEO",
        "genesets_path": genesets_path,
        "cache_dir": tmp_path / "data/processed/external/geo_gse2034",
    }


def test_resolve_training_hyperparameters_selects_elastic_net_ensemble():
    config = {
        "propagation": {"alpha_grid": [0.05, 0.15]},
        "cox": {
            "model_type": "elastic_net_ensemble",
            "penalizer_range": [0.01, 0.1],
            "l1_ratio_range": [0.5, 0.9],
            "n_folds": 5,
            "cv_repeats": 3,
            "min_selection_frequency": 0.6,
            "batch_size": 128,
        },
    }

    resolved = main.resolve_training_hyperparameters(config)

    assert resolved["model_type"] == "elastic_net_ensemble"
    assert resolved["alpha_grid"] == [0.0, 0.05, 0.15]
    assert resolved["penalizer_grid"] == [0.01, 0.1]
    assert resolved["l1_ratio_grid"] == [0.5, 0.9]
    assert resolved["n_folds"] == 5
    assert resolved["cv_repeats"] == 3
    assert resolved["min_selection_frequency"] == 0.6
    assert resolved["batch_size"] == 128
