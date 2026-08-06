"""
MuDAG-Pro independent test-set evaluation pipeline (Sections 4.1 and 4.2 of the paper).

Performs deterministic online inference on the TCGA-BRCA internal test set (174 cases)
and external independent validation cohorts (METABRIC, SCAN-B, and GEO):

1. Load trained Ridge-Cox model weights and optimal hyperparameters.
2. Perform personalized graph propagation (automatically fall back to population-level static propagation if mutation data is missing).
3. Compute C-index prognostic discrimination.
4. Export a CSV of each patient's prognostic index (PI) for subsequent KM/DCA/SHAP analysis.

Key design principles:
- Strictly prevent data leakage: the test set inherits gene_means and gene_stds from the training set.
- Deterministic, hallucination-free online inference: no LLM participates in evaluation.
- Backward compatibility: cohorts without mutation data (SCAN-B and GEO) automatically fall back to static graph propagation.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from lifelines.utils import concordance_index

from src.dataset import PathwayDataset, load_pathway_genesets, load_mutation_signatures
from src.baseline_dag_builder import BaselineDAGBuilder, load_reactome_hierarchy
from src.personalized_graph import PersonalizedGraphModulator
from src.graph_propagation import DirectedPathwayPropagation
from src.cox_model import RidgeCoxSurvivalModel
from src.cox_ensemble import predict_cox_ensemble
from src.pathway_scoring import RobustPathwayScaler
from pipeline.cohort_manifest import freeze_external_cohort
from pipeline.train_tcga import (
    build_propagated_features,
    load_patient_expression_data,
    load_tcga_clinical_data,
    prepare_tcga_split,
)


def validate_checkpoint_tcga_split(checkpoint: Dict, manifest: Dict) -> None:
    """Reject checkpoints trained with a different or undocumented TCGA split."""
    checkpoint_split = (
        checkpoint.get("tcga_split") if isinstance(checkpoint, dict) else None
    )
    required_keys = (
        "version",
        "seed",
        "n_total",
        "n_train",
        "n_test",
        "total_events",
        "train_events",
        "test_events",
    )
    if not isinstance(checkpoint_split, dict):
        raise ValueError(
            "Checkpoint has no verified TCGA split metadata; retrain the model "
            "with the configured 694/174 split."
        )
    mismatches = [
        key
        for key in required_keys
        if checkpoint_split.get(key) != manifest.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint TCGA split does not match the configured manifest "
            f"({', '.join(mismatches)}); retrain the model."
        )


def load_evaluation_data(
    expression_path: str,
    clinical_path: str,
    clinical_split: Optional[str] = None,
    tcga_split_manifest_path: Optional[str] = None,
    tcga_n_train: int = 694,
    tcga_n_test: int = 174,
    tcga_split_seed: int = 42,
    tcga_expected_events: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align expression and survival data for evaluation."""
    if clinical_split is None:
        expression = pd.read_csv(expression_path, index_col=0)
        clinical = pd.read_csv(clinical_path, index_col=0)
    else:
        expression = load_patient_expression_data(expression_path)
        clinical = load_tcga_clinical_data(
            clinical_path,
            split=None,
        )
        tcga_split = prepare_tcga_split(
            expression,
            clinical,
            manifest_path=tcga_split_manifest_path,
            n_train=tcga_n_train,
            n_test=tcga_n_test,
            seed=tcga_split_seed,
            expected_events=tcga_expected_events,
        )
        if clinical_split == "train":
            return tcga_split.train_expression, tcga_split.train_clinical
        if clinical_split == "test":
            return tcga_split.test_expression, tcga_split.test_clinical
        raise ValueError("clinical_split must be 'train', 'test', or None.")

    common_samples = expression.index.intersection(clinical.index)
    if common_samples.empty:
        raise ValueError(
            "Expression and clinical data have no overlapping patient IDs."
        )

    return expression.loc[common_samples], clinical.loc[common_samples]


def build_evaluation_dataset(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    mutations: dict,
    genesets: dict,
    train_ds_meta: PathwayDataset,
    precomputed_pathways: bool = False,
    pathway_scoring_method: str = "zscore_mean",
) -> PathwayDataset:
    """Build an evaluation dataset under the requested feature contract."""
    return PathwayDataset(
        expression,
        genesets,
        clinical,
        mutations,
        is_train=False,
        train_gene_means=train_ds_meta.gene_means,
        train_gene_stds=train_ds_meta.gene_stds,
        precomputed_pathways=precomputed_pathways,
        pathway_scoring_method=pathway_scoring_method,
        train_pathway_scaler=(
            getattr(train_ds_meta, "pathway_scaler", None)
            if pathway_scoring_method == "rank"
            else None
        ),
    )


def bootstrap_cindex(
    times: np.ndarray,
    events: np.ndarray,
    risk: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute an outcome-reporting bootstrap CI without changing predictions."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if not (times.shape == events.shape == risk.shape) or times.ndim != 1:
        raise ValueError("times, events, and risk must be aligned vectors.")
    if len(times) < 2 or n_bootstrap < 1:
        raise ValueError("Bootstrap C-index requires samples and resamples.")

    point_estimate = concordance_index(times, -risk, events)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(times), size=len(times))
        try:
            estimates.append(
                concordance_index(
                    times[indices], -risk[indices], events[indices]
                )
            )
        except ZeroDivisionError:
            continue
    if not estimates:
        raise ValueError("No valid bootstrap concordance samples were generated.")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "c_index": float(point_estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_valid_bootstrap": len(estimates),
    }


def evaluate_dataset(
    dataset_name: str,
    test_expr_path: str,
    test_clin_path: str,
    test_mut_path: Optional[str],
    train_ds_meta: PathwayDataset,
    genesets: dict,
    rules_handbook: list,
    gene2pathway: dict,
    A_enhanced: np.ndarray,
    model_path: str,
    params_path: str,
    clinical_split: Optional[str] = None,
    tcga_split_manifest_path: Optional[str] = None,
    tcga_n_train: int = 694,
    tcga_n_test: int = 174,
    tcga_split_seed: int = 42,
    tcga_expected_events: Optional[int] = None,
    precomputed_pathways: bool = False,
    fixed_cohort: Optional[Dict] = None,
    output_dir: str = "outputs/reports",
) -> Dict:
    """
    Perform deterministic online inference on an independent test set (Sections 4.1 and 4.2 of the paper).

    Complete workflow:
    1. Load optimal hyperparameters and trained model weights.
    2. Build the test-set Dataset (inheriting training-set normalization parameters).
    3. Synthesize patient-specific A_i matrices in a batch and perform graph propagation.
    4. Compute the PI prognostic index.
    5. Compute the C-index and export results.

    Args:
        dataset_name: Dataset name ("TCGA-Holdout", "METABRIC", "SCAN-B", or "GEO").
        test_expr_path: Path to the test-set expression-profile CSV.
        test_clin_path: Path to the test-set clinical-data CSV.
        test_mut_path: Path to the test-set mutation-signature JSON (may be None).
        train_ds_meta: Training-set PathwayDataset (used to extract normalization parameters).
        genesets: Dictionary of 331 pathway gene sets.
        rules_handbook: Global rule handbook R_1.
        gene2pathway: Gene-to-pathway mapping.
        A_enhanced: Enhanced DAG adjacency matrix A (M x M).
        model_path: Path to the .pth model weights.
        params_path: Path to the optimal-hyperparameter JSON.
        output_dir: Result output directory.

    Returns:
        results: Dictionary containing evaluation results such as C-index and PI.
    """
    os.makedirs(output_dir, exist_ok=True)
    pathway_names = list(genesets.keys())
    M = len(pathway_names)

    print(f"\n{'=' * 50}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'=' * 50}")

    # ================================================================
    # 1. Load optimal hyperparameters and trained model weights.
    # ================================================================
    with open(params_path, "r") as f:
        best_params = json.load(f)

    alpha = best_params.get("alpha", 0.10)
    l2_reg = best_params.get("l2_reg", 1e-3)

    # Support two checkpoint formats:
    # - Full checkpoint: {"model_state_dict": ..., "best_lambda": ...}.
    # - Weights only: OrderedDict.
    try:
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    if tcga_split_manifest_path:
        with open(tcga_split_manifest_path, "r", encoding="utf-8") as f:
            tcga_split_manifest = json.load(f)
        validate_checkpoint_tcga_split(checkpoint, tcga_split_manifest)
    is_ensemble = (
        isinstance(checkpoint, dict)
        and checkpoint.get("model_type") == "elastic_net_cox_ensemble_v1"
    )
    model = None
    if is_ensemble:
        alpha = float(checkpoint["alpha"])
        if checkpoint.get("pathway_names") != pathway_names:
            raise ValueError(
                "Checkpoint pathways do not match the evaluation gene sets."
            )
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        alpha = checkpoint.get("alpha", alpha)
        l2_reg = checkpoint.get("best_lambda", checkpoint.get("l2_reg", l2_reg))
    else:
        state_dict = checkpoint

    if not is_ensemble:
        model = RidgeCoxSurvivalModel(in_features=M, l2_reg=l2_reg)
        model.load_state_dict(state_dict)
        model.eval()

    propagator = DirectedPathwayPropagation(alpha=alpha)
    modulator = PersonalizedGraphModulator(
        pathway_names, rules_handbook, gene2pathway
    )

    if is_ensemble:
        print(
            f"  alpha={alpha:.2f}, "
            f"penalizer={checkpoint['best_penalizer']:.3g}, "
            f"l1_ratio={checkpoint['best_l1_ratio']:.2f}, "
            f"members={len(checkpoint['members'])}"
        )
    else:
        print(f"  α = {alpha:.2f}, λ = {l2_reg:.0e}")

    # ================================================================
    # 2. Read test data and build the Dataset.
    # ================================================================
    test_expr, test_clin = load_evaluation_data(
        test_expr_path,
        test_clin_path,
        clinical_split=clinical_split,
        tcga_split_manifest_path=tcga_split_manifest_path,
        tcga_n_train=tcga_n_train,
        tcga_n_test=tcga_n_test,
        tcga_split_seed=tcga_split_seed,
        tcga_expected_events=tcga_expected_events,
    )
    if fixed_cohort:
        frozen = freeze_external_cohort(
            test_expr,
            test_clin,
            manifest_path=fixed_cohort["manifest_path"],
            dataset=dataset_name,
            n_events=int(fixed_cohort["n_events"]),
            n_censored=int(fixed_cohort["n_censored"]),
        )
        test_expr, test_clin = frozen.expression, frozen.clinical
        print(
            f"  Frozen cohort manifest: {fixed_cohort['manifest_path']}"
        )

    # Load mutation signatures when available.
    test_muts = {}
    if test_mut_path and os.path.exists(test_mut_path):
        test_muts = load_mutation_signatures(test_mut_path)
        print(f"  Loaded mutations for {len(test_muts)} patients")
    else:
        print(f"  No mutation data available — model will use static DAG propagation")

    # Build the test-set Dataset.
    # Critical: inherit gene_means and gene_stds from the training set to strictly prevent data leakage.
    test_ds = build_evaluation_dataset(
        expression=test_expr,
        clinical=test_clin,
        mutations=test_muts,
        genesets=genesets,
        train_ds_meta=train_ds_meta,
        precomputed_pathways=precomputed_pathways,
        pathway_scoring_method=("rank" if is_ensemble else "zscore_mean"),
    )

    n_samples = len(test_ds)
    n_events = int(test_ds.events.sum())
    print(f"  Samples: {n_samples}, Events: {n_events}")

    # ================================================================
    # 3. Deterministic inference. Graph matrices are generated in bounded
    # chunks; alpha=0 bypasses graph construction entirely.
    # ================================================================
    x_tilde = build_propagated_features(
        propagator,
        modulator,
        A_enhanced,
        test_ds,
        batch_size=64,
        device="cpu",
    )

    # Compute scalar prognostic index PI_i = β*^T · x̃_i^T.
    if is_ensemble:
        pi = predict_cox_ensemble(
            pd.DataFrame(
                x_tilde.detach().cpu().numpy(),
                index=test_ds.samples,
                columns=pathway_names,
            ),
            checkpoint,
        )
    else:
        with torch.no_grad():
            pi = model(x_tilde).squeeze().cpu().numpy()

    # ================================================================
    # 4. Compute the C-index (Section 4.2 of the paper).
    # ================================================================
    times = test_ds.times.numpy()
    events = test_ds.events.numpy()

        # lifelines concordance_index expects higher risk scores → shorter survival times.
        # Higher PI means higher risk, so pass -pi.
    cindex_report = bootstrap_cindex(
        times, events, pi, n_bootstrap=1000, seed=42
    )
    c_index = cindex_report["c_index"]

    print(
        f"  C-index = {c_index:.4f} "
        f"(95% CI {cindex_report['ci_lower']:.4f}-"
        f"{cindex_report['ci_upper']:.4f})"
    )

    # ================================================================
    # 5. Export the prognostic-index (PI) CSV file.
    # ================================================================
    results_df = pd.DataFrame({
        "sample_id": test_ds.samples,
        "predicted_pi": pi,
        "time": times,
        "event": events,
    })

    # Add risk groups (split at the median).
    pi_median = np.median(pi)
    results_df["risk_group"] = np.where(
        results_df["predicted_pi"] >= pi_median, "High", "Low"
    )

    csv_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Predictions saved to: {csv_path}")

    return {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_events": n_events,
        "c_index": float(c_index),
        "c_index_ci_lower": cindex_report["ci_lower"],
        "c_index_ci_upper": cindex_report["ci_upper"],
        "n_valid_bootstrap": cindex_report["n_valid_bootstrap"],
        "pi_median": float(pi_median),
        "predictions_csv": csv_path,
        "results_df": results_df,
    }


def evaluate_all_cohorts(
    train_expr_path: str,
    train_clin_path: str,
    train_mut_path: str,
    genesets_path: str,
    reactome_path: str,
    rules_path: str,
    gene2pathway_path: str,
    model_path: str,
    params_path: str,
    test_cohorts: List[Dict],
    tcga_split_manifest_path: Optional[str] = None,
    tcga_n_train: int = 694,
    tcga_n_test: int = 174,
    tcga_split_seed: int = 42,
    tcga_expected_events: Optional[int] = None,
    output_dir: str = "outputs/reports",
) -> Dict[str, Dict]:
    """
    Evaluate all test cohorts in a batch (Section 4.2 of the paper).

    Args:
        train_expr_path: Path to the training-set expression profile.
        train_clin_path: Path to the training-set clinical data.
        train_mut_path: Path to the training-set mutation signatures.
        genesets_path: Path to the pathway gene sets.
        reactome_path: Path to the Reactome hierarchy.
        rules_path: Path to the rule handbook.
        gene2pathway_path: Path to the gene-to-pathway mapping.
        model_path: Path to the model weights.
        params_path: Path to the hyperparameters.
        test_cohorts: List of test cohorts, each containing:
            {"name": "METABRIC", "expr": "...", "clinical": "...", "mutations": "..."}
        output_dir: Output directory.

    Returns:
        all_results: {dataset_name: result_dict}
    """
    # Load shared prior knowledge.
    print("=" * 60)
    print("MuDAG-Pro Multi-Cohort Evaluation (Section 4.2 of the paper)")
    print("=" * 60)

    print("\nLoading prior knowledge...")
    genesets = load_pathway_genesets(genesets_path)
    pathway_names = list(genesets.keys())

    parent_child_pairs = load_reactome_hierarchy(reactome_path)

    with open(rules_path, "r", encoding="utf-8") as f:
        rules_handbook = json.load(f)

    with open(gene2pathway_path, "r", encoding="utf-8") as f:
        gene2pathway = json.load(f)

    # Build the enhanced DAG.
    dag_builder = BaselineDAGBuilder(pathway_names)
    _, A_0 = dag_builder.build_baseline_dag(parent_child_pairs)
    A_enhanced, _ = dag_builder.build_enhanced_dag(A_0, rules_handbook)

    # Build the training-set Dataset (to expose normalization parameters).
    print("\nBuilding training dataset for normalization parameters...")
    train_expr, train_clin = load_evaluation_data(
        train_expr_path,
        train_clin_path,
        clinical_split="train",
        tcga_split_manifest_path=tcga_split_manifest_path,
        tcga_n_train=tcga_n_train,
        tcga_n_test=tcga_n_test,
        tcga_split_seed=tcga_split_seed,
        tcga_expected_events=tcga_expected_events,
    )
    train_muts = load_mutation_signatures(train_mut_path)
    try:
        shared_checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        shared_checkpoint = torch.load(model_path, map_location="cpu")
    is_ensemble = (
        isinstance(shared_checkpoint, dict)
        and shared_checkpoint.get("model_type")
        == "elastic_net_cox_ensemble_v1"
    )
    if is_ensemble:
        pathway_scaler = RobustPathwayScaler.from_dict(
            shared_checkpoint["pathway_scaler"]
        )
        train_ds = PathwayDataset(
            train_expr,
            genesets,
            train_clin,
            train_muts,
            is_train=False,
            pathway_scoring_method="rank",
            train_pathway_scaler=pathway_scaler,
        )
        print(
            f"  Training reference: {len(train_ds)} samples, "
            f"{len(pathway_scaler.medians)} rank pathways scaled"
        )
    else:
        train_ds = PathwayDataset(
            train_expr, genesets, train_clin, train_muts,
            is_train=True,
        )
        print(f"  Training reference: {len(train_ds)} samples, "
              f"{len(train_ds.gene_means)} genes normalized")

    # Evaluate each test cohort in turn.
    all_results = {}

    for cohort in test_cohorts:
        name = cohort["name"]
        result = evaluate_dataset(
            dataset_name=name,
            test_expr_path=cohort["expr"],
            test_clin_path=cohort["clinical"],
            test_mut_path=cohort.get("mutations"),
            train_ds_meta=train_ds,
            genesets=genesets,
            rules_handbook=rules_handbook,
            gene2pathway=gene2pathway,
            A_enhanced=A_enhanced,
            model_path=model_path,
            params_path=params_path,
            clinical_split=cohort.get("clinical_split"),
            tcga_split_manifest_path=tcga_split_manifest_path,
            tcga_n_train=tcga_n_train,
            tcga_n_test=tcga_n_test,
            tcga_split_seed=tcga_split_seed,
            tcga_expected_events=tcga_expected_events,
            precomputed_pathways=cohort.get("precomputed_pathways", False),
            fixed_cohort=cohort.get("fixed_cohort"),
            output_dir=output_dir,
        )
        all_results[name] = result

    # Print the summary table.
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"{'Cohort':<20s} {'Samples':<10s} {'Events':<10s} {'C-index':<10s}")
    print("-" * 50)
    for name, res in all_results.items():
        print(f"{name:<20s} {res['n_samples']:<10d} "
              f"{res['n_events']:<10d} {res['c_index']:<10.4f}")

    # Save the summary JSON.
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    summary = {
        name: {k: v for k, v in res.items() if k != "results_df"}
        for name, res in all_results.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MuDAG-Pro Multi-Cohort Evaluation"
    )
    parser.add_argument(
        "--train_expr", type=str,
        default="data/raw/tcga_brca/expr_train.csv",
    )
    parser.add_argument(
        "--train_clinical", type=str,
        default="data/raw/tcga_brca/clinical_train.csv",
    )
    parser.add_argument(
        "--train_mutations", type=str,
        default="data/raw/tcga_brca/mutations.json",
    )
    parser.add_argument(
        "--genesets", type=str,
        default="data/processed/pathway_genesets/331_pathways.json",
    )
    parser.add_argument(
        "--reactome", type=str,
        default="data/raw/reactome/hierarchy.json",
    )
    parser.add_argument(
        "--rules", type=str,
        default="knowledge_engine/rules_handbook.json",
    )
    parser.add_argument(
        "--gene2pathway", type=str,
        default="data/processed/gene2pathway.json",
    )
    parser.add_argument(
        "--model", type=str,
        default="outputs/models/mudag_pro_model.pth",
    )
    parser.add_argument(
        "--params", type=str,
        default="outputs/models/best_params.json",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/reports",
    )

    args = parser.parse_args()

    # Define test cohorts.
    test_cohorts = [
        {
            "name": "TCGA-Holdout",
            "expr": "data/raw/tcga_brca/expr_test.csv",
            "clinical": "data/raw/tcga_brca/clinical_test.csv",
            "mutations": "data/raw/tcga_brca/mutations_test.json",
        },
        {
            "name": "METABRIC",
            "expr": "data/raw/metabric/expression.csv",
            "clinical": "data/raw/metabric/clinical.csv",
            "mutations": "data/raw/metabric/mutations.json",
        },
        {
            "name": "SCAN-B",
            "expr": "data/raw/scan_b/expression.csv",
            "clinical": "data/raw/scan_b/clinical.csv",
            "mutations": None,  # SCAN-B has no mutation data.
        },
        {
            "name": "GEO",
            "expr": "data/raw/geo_gse2034/expression.csv",
            "clinical": "data/raw/geo_gse2034/clinical.csv",
            "mutations": None,  # GEO has no mutation data.
        },
    ]

    evaluate_all_cohorts(
        train_expr_path=args.train_expr,
        train_clin_path=args.train_clinical,
        train_mut_path=args.train_mutations,
        genesets_path=args.genesets,
        reactome_path=args.reactome,
        rules_path=args.rules,
        gene2pathway_path=args.gene2pathway,
        model_path=args.model,
        params_path=args.params,
        test_cohorts=test_cohorts,
        output_dir=args.output_dir,
    )
