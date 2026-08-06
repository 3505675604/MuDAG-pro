import json

import numpy as np
import pandas as pd

from src.dataset import PathwayDataset, load_mutation_signatures


def test_load_mutation_signatures_merges_tcga_aliquots_by_patient(tmp_path):
    mutation_path = tmp_path / "mutations.json"
    mutation_path.write_text(
        json.dumps(
            {
                "tcga-ab-1234-01A": [
                    ["pik3ca", "missense_variant"],
                    ["TP53", "stop_gained"],
                ],
                "TCGA-AB-1234-02A": [
                    ["PIK3CA", "missense_variant"],
                    ["CDH1", "frameshift_variant"],
                ],
                "METABRIC-001": [["ERBB2", "amplification"]],
            }
        ),
        encoding="utf-8",
    )

    mutations = load_mutation_signatures(str(mutation_path))

    assert mutations == {
        "TCGA-AB-1234": [
            ("PIK3CA", "missense_variant"),
            ("TP53", "stop_gained"),
            ("CDH1", "frameshift_variant"),
        ],
        "METABRIC-001": [("ERBB2", "amplification")],
    }


def test_pathway_dataset_uses_shared_rank_scores_and_training_scaler():
    expression = pd.DataFrame(
        {
            "G1": [1.0, 8.0, 1.0],
            "G2": [2.0, 4.0, 4.0],
            "G3": [4.0, 2.0, 2.0],
            "G4": [8.0, 1.0, 8.0],
        },
        index=["S1", "S2", "S3"],
    )
    clinical = pd.DataFrame(
        {"time": [1.0, 2.0, 3.0], "event": [1.0, 0.0, 1.0]},
        index=expression.index,
    )
    genesets = {"LOW": ["G1", "G2"], "HIGH": ["G3", "G4"]}

    training = PathwayDataset(
        expression,
        genesets,
        clinical,
        pathway_scoring_method="rank",
        is_train=True,
    )
    external = PathwayDataset(
        np.exp(expression),
        genesets,
        clinical,
        pathway_scoring_method="rank",
        is_train=False,
        train_pathway_scaler=training.pathway_scaler,
    )

    pd.testing.assert_frame_equal(
        external.pathway_scores_df,
        training.pathway_scores_df,
    )
