"""
MuDAG-Pro project entry point.

Coordinates data preprocessing, model training, inference evaluation, ablation experiments, and report generation.

Usage:
    python main.py --mode preprocess        # Preprocess data (generate .npy/.json from raw CSV files).
    python main.py --mode train             # 5-fold cross-validation + grid-search training.
    python main.py --mode evaluate          # Evaluate independent test sets.
    python main.py --mode ablation          # Run ablation experiments.
    python main.py --mode knowledge_refine  # Refine knowledge offline (LLM + RAG).
    python main.py --mode report            # Generate personalized white-box reports.
"""
import os
import sys
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Optional, List

from pipeline.geo_gse2034_adapter import prepare_geo_gse2034
from pipeline.metabric_adapter import prepare_metabric
from pipeline.scan_b_adapter import prepare_scan_b

def setup_environment(config_path: str = "config/base_config.yaml") -> Dict:
    """Load configuration, set random seeds, and create output directories."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    for dir_key in ["models_dir", "figures_dir", "reports_dir"]:
        os.makedirs(config["output"][dir_key], exist_ok=True)

    return config


def mode_preprocess(config: Dict, args):
    """
    Complete data-preprocessing pipeline.

    Generate all preprocessed data from raw CSV files:
    - Activation-score matrix for 331 pathways (N×331).
    - DAG adjacency matrix (transitive closure + enhancement).
    - Gene-to-pathway mapping and rule handbook.
    - Survival data and mutation signatures.
    """
    print("=" * 70)
    print("MuDAG-Pro Data Preprocessing (331 Pathways)")
    print("=" * 70)

    # Delegate to the main function in preprocess_data.py.
    import preprocess_data
    preprocess_data.main()


def resolve_training_hyperparameters(config: Dict) -> Dict:
    """Translate YAML training settings into pipeline keyword arguments."""
    propagation_config = config.get("propagation", {})
    cox_config = config.get("cox", {})

    configured_alphas = propagation_config.get(
        "alpha_grid", [propagation_config.get("alpha", 0.15)]
    )
    alpha_grid = sorted({0.0, *(float(a) for a in configured_alphas)})

    return {
        "model_type": str(cox_config.get("model_type", "ridge_cox")),
        "alpha_grid": alpha_grid,
        "l2_reg_grid": [
            float(value)
            for value in cox_config.get(
                "lambda_range", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
            )
        ],
        "penalizer_grid": [
            float(value)
            for value in cox_config.get(
                "penalizer_range", [0.001, 0.01, 0.05, 0.1, 0.5]
            )
        ],
        "l1_ratio_grid": [
            float(value)
            for value in cox_config.get("l1_ratio_range", [0.1, 0.5, 0.9])
        ],
        "n_folds": int(cox_config.get("n_folds", 5)),
        "cv_repeats": int(cox_config.get("cv_repeats", 3)),
        "min_selection_frequency": float(
            cox_config.get("min_selection_frequency", 0.6)
        ),
        "learning_rate": float(cox_config.get("learning_rate", 1e-3)),
        "max_epochs": int(cox_config.get("max_epochs", 500)),
        "patience": int(cox_config.get("early_stopping_patience", 50)),
        "batch_size": int(cox_config.get("batch_size", 64)),
    }

def mode_train(config: Dict, args):
    """
    TCGA-BRCA 5-fold cross-validation training (Section 4.1 of the paper).

    Read data from raw CSV files and perform:
    1. Pathway activation-score calculation.
    2. DAG construction (gene-overlap inference).
    3. Grid search over α × λ.
    4. Retraining on the full training set.
    5. Saving the model checkpoint.
    """
    from pipeline.train_tcga import run_5fold_cv_training

    data_config = config["data"]
    pathway_config = config["pathways"]
    tcga_config = data_config["tcga_brca"]

    # Resolve paths.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    expr_path = os.path.join(base_dir, tcga_config["expression_file"])
    clinical_path = os.path.join(base_dir, tcga_config["clinical_file"])
    mut_path = os.path.join(base_dir, tcga_config["mutation_file"])
    split_manifest_path = os.path.join(
        base_dir, tcga_config["split_manifest_file"]
    )
    genesets_path = os.path.join(
        base_dir, pathway_config.get(
            "geneset_file",
            "data/processed/pathway_genesets/pathway_genesets.json"
        )
    )
    reactome_path = os.path.join(
        base_dir, "data/processed/pathway_genesets/reactome_hierarchy.json"
    )

    # Use preprocessed files when they already exist.
    if os.path.exists(genesets_path) and os.path.exists(reactome_path):
        print("[Train] Using preprocessed data...")
    else:
        print("[Train] Preprocessed data not found, will compute dynamically from raw CSV...")
        genesets_path = os.path.join(
            base_dir, "datasets_csv/metadata/combine_signatures.csv"
        )
        reactome_path = os.path.join(
            base_dir, "data/processed/pathway_genesets/reactome_hierarchy.json"
        )
    # If the Reactome hierarchy does not yet exist, build_dag_from_reactome will generate it.

    rules_path = os.path.join(base_dir, "knowledge_engine/rules_handbook.json")
    gene2pathway_path = os.path.join(
        base_dir, "data/processed/pathway_genesets/gene_to_pathway.json"
    )

    output_dir = os.path.join(base_dir, config["output"]["models_dir"])

    results = run_5fold_cv_training(
        expr_path=expr_path,
        clinical_path=clinical_path,
        genesets_path=genesets_path,
        reactome_path=reactome_path,
        rules_path=rules_path,
        gene2pathway_path=gene2pathway_path,
        mut_path=mut_path,
        output_dir=output_dir,
        seed=getattr(args, 'seed', 42),
        device=getattr(args, 'device', 'cpu'),
        n_train=int(tcga_config["n_train"]),
        n_test=int(tcga_config["n_test"]),
        split_seed=int(tcga_config.get("split_seed", 42)),
        split_manifest_path=split_manifest_path,
        expected_events=int(tcga_config["n_events"]),
        **resolve_training_hyperparameters(config),
    )

    return results



def resolve_custom_evaluation_cohorts(
    dataset_names: List[str],
    base_dir: str,
    genesets_path: str,
) -> List[Dict]:
    """Resolve requested external cohorts to model-ready indexed artifacts."""
    normalized_names = []
    for value in dataset_names:
        normalized_names.extend(
            name.strip().lower()
            for name in value.split(",")
            if name.strip()
        )

    cohorts = []
    for dataset_name in normalized_names:
        if dataset_name == "metabric":
            prepared = prepare_metabric(
                raw_dir=os.path.join(
                    base_dir,
                    "Data/METABRIC/metabric_raw/brca_metabric",
                ),
                genesets_path=genesets_path,
                cache_dir=os.path.join(
                    base_dir,
                    "data/processed/external/metabric",
                ),
            )
            cohorts.append({
                "name": "METABRIC",
                **prepared,
                "fixed_cohort": {
                    "manifest_path": os.path.join(
                        base_dir,
                        "data/processed/splits/"
                        "metabric_1413_820_sha256.json",
                    ),
                    "n_events": 820,
                    "n_censored": 593,
                },
            })
        elif dataset_name in {"scan_b", "scan-b"}:
            prepared = prepare_scan_b(
                raw_dir=os.path.join(base_dir, "Data/SCAN-B"),
                genesets_path=genesets_path,
                cache_dir=os.path.join(
                    base_dir,
                    "data/processed/external/scan_b",
                ),
            )
            cohorts.append({
                "name": "SCAN-B",
                **prepared,
                "fixed_cohort": {
                    "manifest_path": os.path.join(
                        base_dir,
                        "data/processed/splits/scan_b_2449_219_sha256.json",
                    ),
                    "n_events": 219,
                    "n_censored": 2230,
                },
            })
        elif dataset_name in {"geo_gse2034", "geo-gse2034", "gse2034"}:
            prepared = prepare_geo_gse2034(
                raw_dir=os.path.join(base_dir, "Data/GEO"),
                genesets_path=genesets_path,
                cache_dir=os.path.join(
                    base_dir,
                    "data/processed/external/geo_gse2034",
                ),
            )
            cohorts.append({"name": "GEO-GSE2034", **prepared})
        else:
            raise ValueError(
                f"Unsupported external dataset: {dataset_name}. "
                "Currently supported: metabric, scan_b, geo_gse2034."
            )

    return cohorts


def mode_evaluate(config: Dict, args):
    """
    Multicenter independent-cohort evaluation (Section 4.2 of the paper).

    Evaluate model performance on TCGA-Holdout, METABRIC, SCAN-B, and GEO.
    Outputs: C-index, AUC(t), PI prediction CSV, and KM curve data.
    """
    from pipeline.evaluate import evaluate_all_cohorts

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_config = config["data"]
    tcga_config = data_config["tcga_brca"]
    output_dir = os.path.join(base_dir, config["output"]["reports_dir"])

    # Training-set paths (used to expose normalization parameters).
    train_expr = os.path.join(base_dir, tcga_config["expression_file"])
    train_clin = os.path.join(base_dir, tcga_config["clinical_file"])
    train_mut = os.path.join(base_dir, tcga_config["mutation_file"])
    split_manifest_path = os.path.join(
        base_dir, tcga_config["split_manifest_file"]
    )

    # Model paths.
    model_path = getattr(args, 'checkpoint', None) or os.path.join(
        base_dir, config["output"]["models_dir"], "best_model.pt"
    )
    params_path = os.path.join(
        base_dir, config["output"]["models_dir"], "best_params.json"
    )

    # Prior-knowledge paths.
    genesets_path = os.path.join(
        base_dir, "data/processed/pathway_genesets/pathway_genesets.json"
    )
    reactome_path = os.path.join(
        base_dir, "data/processed/pathway_genesets/reactome_hierarchy.json"
    )
    rules_path = os.path.join(base_dir, "knowledge_engine/rules_handbook.json")
    gene2pathway_path = os.path.join(
        base_dir, "data/processed/pathway_genesets/gene_to_pathway.json"
    )

    # Check preprocessed data.
    if not os.path.exists(genesets_path):
        print("[Evaluate] Preprocessed data not found, please run: python main.py --mode preprocess")
        return

    # Build the list of test cohorts.
    test_cohorts = [
        {
            "name": "TCGA-Holdout",
            "expr": train_expr,
            "clinical": train_clin,  # Use the same clinical data and filter by test_idx.
            "mutations": train_mut,
            "clinical_split": "test",
        },
        # METABRIC/SCAN-B/GEO require separate preprocessing; placeholders are used for now.
    ]

    # If the user specified a dataset.
    if getattr(args, 'datasets', None):
        test_cohorts = resolve_custom_evaluation_cohorts(
            args.datasets,
            base_dir=base_dir,
            genesets_path=genesets_path,
        )

    results = evaluate_all_cohorts(
        train_expr_path=train_expr,
        train_clin_path=train_clin,
        train_mut_path=train_mut,
        genesets_path=genesets_path,
        reactome_path=reactome_path,
        rules_path=rules_path,
        gene2pathway_path=gene2pathway_path,
        model_path=model_path,
        params_path=params_path,
        test_cohorts=test_cohorts,
        tcga_split_manifest_path=split_manifest_path,
        tcga_n_train=int(tcga_config["n_train"]),
        tcga_n_test=int(tcga_config["n_test"]),
        tcga_split_seed=int(tcga_config.get("split_seed", 42)),
        tcga_expected_events=int(tcga_config["n_events"]),
        output_dir=output_dir,
    )

    return results


# ================================================================
# 4. Ablation experiments.
# ================================================================

def mode_ablation(config: Dict, args):
    """
    Ablation experiments (Section 4.3 of the paper).

    Run five variants:
    - M_static: No personalized edge-weight adjustment.
    - M_onehot: Assign a fixed constant to all mutation-pathway edge weights.
    - M_rand_dist: Randomly shuffle mutation-to-edge mappings.
    - M_inter: Pathway × mutation interaction terms.
    - M_LOO: Leave-One-Out edge-by-edge ablation.
    """
    from pipeline.run_ablations import run_ablations

    base_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = getattr(args, 'checkpoint', None) or os.path.join(
        base_dir, config["output"]["models_dir"], "best_model.pt"
    )
    data_dir = os.path.join(base_dir, config["data"]["processed_dir"])
    output_dir = os.path.join(base_dir, config["output"]["models_dir"]).replace(
        "/models", ""
    )

    variants = getattr(args, 'variants', None)
    seed = getattr(args, 'seed', 42)
    device = getattr(args, 'device', 'cpu')

    results = run_ablations(
        checkpoint_path=checkpoint_path,
        config_path=os.path.join(base_dir, "config/base_config.yaml"),
        data_dir=data_dir,
        output_dir=output_dir,
        variants=variants,
        seed=seed,
        device=device,
    )

    return results


# ================================================================
# 5. Offline knowledge refinement.
# ================================================================

def mode_knowledge_refine(config: Dict, args):
    """
    Offline LLM + RAG knowledge refinement (Section 3.3 of the paper).

    Retrieve literature evidence from PubMed, use an LLM to parse regulatory relationships
    between pathways, and generate the globally frozen rule handbook R_1.

    Note: Requires an OpenAI API key and PubMed API access.
    """
    from knowledge_engine.pubmed_retriever import PubMedRetriever
    from knowledge_engine.llm_agent import PathwayLLMAgent
    from knowledge_engine.dag_refine_filter import RuleHandbookGenerator

    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("MuDAG-Pro Offline Knowledge Refinement (LLM + RAG, Section 3.3 of the paper)")
    print("=" * 70)

    # Load pathway gene sets.
    geneset_path = os.path.join(
        base_dir, "data/processed/pathway_genesets/pathway_genesets.json"
    )
    if not os.path.exists(geneset_path):
        print("[Knowledge Refine] Preprocessed data not found, please run python main.py --mode preprocess")
        return

    with open(geneset_path, 'r', encoding='utf-8') as f:
        pathway_genesets = json.load(f)
    pathway_names = list(pathway_genesets.keys())
    print(f"  Loaded {len(pathway_names)} pathways")

    # Initialize components.
    llm_config = config["knowledge_engine"]["llm"]
    pubmed_config = config["knowledge_engine"]["pubmed"]

    retriever = PubMedRetriever(
        email=getattr(args, 'pubmed_email', None) or "researcher@example.com",
        api_key=getattr(args, 'ncbi_api_key', None),
    )

    llm_agent = PathwayLLMAgent(
        api_key=getattr(args, 'openai_api_key', None),
        model_name=llm_config.get("model", "gpt-4o"),
    )

    rule_generator = RuleHandbookGenerator(pathway_names=pathway_names)

    # Load baseline DAG edges for cycle detection.
    dag_dir = os.path.join(base_dir, "data/processed/baseline_dag")
    baseline_edges = set()
    if os.path.exists(os.path.join(dag_dir, "baseline_adj.npy")):
        baseline_adj = np.load(os.path.join(dag_dir, "baseline_adj.npy"))
        for u in range(baseline_adj.shape[0]):
            for v in range(baseline_adj.shape[1]):
                if baseline_adj[u, v] > 0:
                    u_name = pathway_names[u] if u < len(pathway_names) else f"P{u}"
                    v_name = pathway_names[v] if v < len(pathway_names) else f"P{v}"
                    baseline_edges.add((u_name, v_name))

    # Disease context (M_context in Section 3.3 of the paper).
    context_terms = [
        "HR+ HER2- breast cancer",
        "endocrine resistance",
        "PIK3CA mutation",
        "ESR1 mutation",
        "CDK4/6 inhibitor",
    ]

    all_rules = []
    print(f"\n  Starting literature mining...")

    if getattr(args, 'interactive', False):
        # Interactive mode: focus on key pathways.
        key_keywords = ["pi3k", "akt", "mtor", "apoptosis", "estrogen",
                        "cell_cycle", "dna_repair", "p53", "mapk"]

        key_pathways = []
        for pw in pathway_names:
            pw_lower = pw.lower()
            for kw in key_keywords:
                if kw in pw_lower:
                    key_pathways.append(pw)
                    break

        key_pathways = list(set(key_pathways))[:20]  # At most 20.
        print(f"  Interactive mode: {len(key_pathways)} key pathways")

        for i, src in enumerate(key_pathways):
            for tgt in key_pathways:
                if src == tgt:
                    continue
                src_genes = pathway_genesets.get(src, [])[:5]
                print(f"  [{i+1}/{len(key_pathways)}] Searching: {src} → {tgt}")

                articles = retriever.search_evidence(
                    source_pathway=src,
                    target_pathway=tgt,
                    context_terms=context_terms,
                    max_results=3,
                )

                if articles:
                    for gene in src_genes[:3]:
                        rule = llm_agent.evaluate_rule(src, tgt, gene, articles)
                        if rule is not None:
                            all_rules.append(rule)

                if len(all_rules) >= 200:
                    break
            if len(all_rules) >= 200:
                break

    # Refine and save the rule handbook.
    rules_handbook = rule_generator.filter_and_export(
        raw_rules=all_rules,
        output_json_path=os.path.join(
            base_dir, "knowledge_engine/rules_handbook.json"
        ),
        baseline_edges=baseline_edges,
    )

    print(f"\n[Knowledge Refine] Done! Total {len(rules_handbook)} rules")
    return rules_handbook


# ================================================================
# 6. Personalized report generation.
# ================================================================

def mode_report(config: Dict, args):
    """
    Generate personalized white-box clinical decision reports (Section 3.7 of the paper).

    Generate physician-readable prognostic reports from trained model weights, patient PI scores,
    key-pathway contribution rankings, and somatic mutation context.
    """
    from analysis.report_generator import ReportGenerator

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the model checkpoint.
    checkpoint_path = getattr(args, 'checkpoint', None) or os.path.join(
        base_dir, config["output"]["models_dir"], "best_model.pt"
    )

    if not os.path.exists(checkpoint_path):
        print(f"[Report] Checkpoint not found: {checkpoint_path}")
        print("[Report] Please run: python main.py --mode train")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Obtain pathway names.
    if "pathway_names" in checkpoint:
        pathway_names = checkpoint["pathway_names"]
    else:
        pathway_names = [f"Pathway_{i}" for i in range(checkpoint["n_features"])]

    # Load the rule handbook.
    rules_path = os.path.join(base_dir, "knowledge_engine/rules_handbook.json")
    rules = []
    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

    generator = ReportGenerator(
        pathway_names=pathway_names,
        rules_handbook=rules,
    )

    # Load prediction results when available.
    preds_path = os.path.join(
        base_dir, config["output"]["reports_dir"], "TCGA-Holdout_predictions.csv"
    )

    if os.path.exists(preds_path):
        preds_df = pd.read_csv(preds_path)
        # Select one high-risk and one low-risk case.
        high_risk = preds_df[preds_df["risk_group"] == "High"].iloc[0]
        low_risk = preds_df[preds_df["risk_group"] == "Low"].iloc[0]

        for patient_name, patient_data in [("HIGH_RISK_EXAMPLE", high_risk),
                                            ("LOW_RISK_EXAMPLE", low_risk)]:
            report = generator.generate_report(
                patient_id=patient_data.get("sample_id", patient_name),
                pi_score=float(patient_data["predicted_pi"]),
                risk_group=patient_data.get("risk_group", "Unknown"),
                risk_percentile=50.0,
                top_pathways=[],  # Must be obtained from the model.
                mutation_signature=[],
                matched_rules=[],
                use_llm=False,
            )
            generator.save_report(
                report, output_dir=config["output"]["reports_dir"]
            )
            print(f"[Report] Generated: {patient_name}")
    else:
    # Generate example reports.
        print("[Report] Prediction data not found, generating example report...")
        report = generator.generate_report(
            patient_id="SAMPLE_001",
            pi_score=0.85,
            risk_group="High Risk",
            risk_percentile=85.0,
            top_pathways=[
                {"pathway": "PI3K/AKT/mTOR", "hr": 2.3, "beta": 0.83,
                 "risk_direction": "risk"},
                {"pathway": "ER Signaling", "hr": 0.45, "beta": -0.80,
                 "risk_direction": "protective"},
            ],
            mutation_signature=[("PIK3CA", "H1047R"), ("ESR1", "Y537S")],
            matched_rules=[],
            use_llm=False,
        )
        generator.save_report(report, output_dir=config["output"]["reports_dir"])
        print(f"[Report] Example report generated")


# ================================================================
# Main entry point.
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MuDAG-Pro: Mutation-contextualized Dynamic Pathway Graph Propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode preprocess
  python main.py --mode train --device cuda:0
  python main.py --mode evaluate
  python main.py --mode ablation --variants M_static M_onehot
  python main.py --mode knowledge_refine --interactive
  python main.py --mode report
        """,
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["preprocess", "train", "evaluate", "ablation",
                 "knowledge_refine", "report"],
        help="Run mode",
    )
    parser.add_argument(
        "--config", type=str, default="config/base_config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device (cpu, cuda:0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="List of evaluation datasets",
    )
    parser.add_argument(
        "--variants", type=str, nargs="+", default=None,
        help="List of ablation variants",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Knowledge refinement: interactive hypothetical edge generation",
    )
    parser.add_argument(
        "--pubmed_email", type=str, default=None,
        help="PubMed API email",
    )
    parser.add_argument(
        "--ncbi_api_key", type=str, default=None,
        help="NCBI API key",
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=None,
        help="OpenAI API key",
    )

    args = parser.parse_args()

    # Load the configuration.
    config = setup_environment(args.config)

    # Dispatch by mode.
    mode_dispatch = {
        "preprocess": mode_preprocess,
        "train": mode_train,
        "evaluate": mode_evaluate,
        "ablation": mode_ablation,
        "knowledge_refine": mode_knowledge_refine,
        "report": mode_report,
    }

    mode_fn = mode_dispatch[args.mode]
    mode_fn(config, args)


if __name__ == "__main__":
    main()
