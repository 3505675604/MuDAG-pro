"""
MuDAG-Pro TCGA-BRCA training pipeline (Section 4.1 of the paper).

Runs on the TCGA-BRCA training set of 694 cases:
1. 5-fold cross-validation.
2. Grid search for the optimal propagation coefficient α and regularization parameter λ.
3. Retraining on the full training set and saving the optimal model weights β* and hyperparameter checkpoint.

Core training workflow:
    for each (alpha, lambda) in grid:
        for each fold in 5-fold CV:
            1. Build PathwayDataset (independent Z-score normalization for training and validation sets).
            2. Synthesize patient-specific adjacency matrices in a batch: A_batch = A ⊙ Γ_i.
            3. Perform single-step directed graph pathway propagation: x̃ = x + α · x · A_i.
            4. Use Ridge-Cox partial-likelihood loss + L2 regularization.
            5. Update parameters through backpropagation.
        Evaluate the validation-set C-index and select the optimal hyperparameters.
"""
import os
import json
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from lifelines.utils import concordance_index

# Import low-level operator modules.
from src.dataset import PathwayDataset, load_pathway_genesets, load_mutation_signatures
from src.baseline_dag_builder import BaselineDAGBuilder, load_reactome_hierarchy
from src.personalized_graph import PersonalizedGraphModulator
from src.graph_propagation import DirectedPathwayPropagation, batch_build_patient_adjacencies
from src.cox_model import RidgeCoxSurvivalModel, train_ridge_cox
from src.cox_ensemble import (
    fit_elastic_net_cox_ensemble,
    predict_cox_ensemble,
)


@dataclass(frozen=True)
class TCGASplit:
    """Aligned TCGA frames and the persisted patient-level split contract."""

    train_expression: pd.DataFrame
    train_clinical: pd.DataFrame
    test_expression: pd.DataFrame
    test_clinical: pd.DataFrame
    manifest: Dict


def build_cv_splits(
    events: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build reproducible folds with balanced event/censoring labels."""
    event_labels = np.asarray(events, dtype=int)
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )
    sample_indices = np.arange(len(event_labels))
    return list(splitter.split(sample_indices, event_labels))


def select_final_training_epochs(
    cv_results: List[Dict],
    best_hyperparams: Dict[str, float],
    max_epochs: int,
) -> int:
    """Transfer the typical CV early-stop duration to full-data fitting."""
    for result in cv_results:
        if (
            np.isclose(result["alpha"], best_hyperparams["alpha"])
            and np.isclose(result["l2_reg"], best_hyperparams["l2_reg"])
        ):
            fold_epochs = result.get("fold_epochs", [])
            if fold_epochs:
                return max(
                    1,
                    min(max_epochs, int(np.median(fold_epochs))),
                )
    return max_epochs


def load_tcga_clinical_data(
    clinical_path: str,
    split: Optional[str] = "train",
) -> pd.DataFrame:
    """Load TCGA clinical data in the patient-level survival schema."""
    raw = pd.read_csv(clinical_path)
    raw_columns = set(raw.columns)

    if {"case_id", "survival_months", "censorship"}.issubset(raw_columns):
        clinical = raw.drop_duplicates(subset="case_id").copy()
        clinical = clinical.set_index("case_id")
        clinical["event"] = 1.0 - clinical["censorship"].astype(float)
        clinical["time"] = clinical["survival_months"].astype(float) * 30.44
    elif {"event", "time"}.issubset(raw_columns):
        clinical = pd.read_csv(clinical_path, index_col=0)
    else:
        raise ValueError(
            "Clinical data must contain either the raw TCGA columns "
            "('case_id', 'survival_months', 'censorship') or the "
            "standardized survival columns ('time', 'event')."
        )

    if split not in {None, "train", "test"}:
        raise ValueError("split must be 'train', 'test', or None.")

    if "train" in clinical.columns and split is not None:
        expected_value = 1.0 if split == "train" else 0.0
        clinical = clinical.loc[
            clinical["train"].astype(float) == expected_value
        ]

    if clinical.index.has_duplicates:
        raise ValueError("Clinical patient IDs must be unique after normalization.")

    return clinical


def load_patient_expression_data(expression_path: str) -> pd.DataFrame:
    """Load one gene-expression profile per patient."""
    expression = pd.read_csv(expression_path, index_col=0)
    if expression.index.has_duplicates:
        expression = expression.groupby(level=0, sort=False).mean()
    return expression


def prepare_tcga_split(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    manifest_path: str,
    n_train: int = 694,
    n_test: int = 174,
    seed: int = 42,
    expected_events: Optional[int] = None,
) -> TCGASplit:
    """Align TCGA patients, then create or reuse one exact stratified split."""
    if n_train <= 0 or n_test <= 0:
        raise ValueError("TCGA train and test sizes must both be positive.")
    if not manifest_path:
        raise ValueError("A TCGA split manifest path is required.")
    if "event" not in clinical.columns:
        raise ValueError("TCGA clinical data must contain an 'event' column.")

    expression = expression.copy()
    clinical = clinical.copy()
    expression.index = expression.index.map(str)
    clinical.index = clinical.index.map(str)
    if expression.index.has_duplicates or clinical.index.has_duplicates:
        raise ValueError("TCGA patient IDs must be unique before splitting.")

    common_ids = sorted(set(expression.index) & set(clinical.index))
    expected_total = n_train + n_test
    if len(common_ids) != expected_total:
        raise ValueError(
            f"aligned TCGA cohort has {len(common_ids)} patients; "
            f"expected exactly {expected_total} ({n_train} train + {n_test} test)."
        )

    aligned_expression = expression.loc[common_ids]
    aligned_clinical = clinical.loc[common_ids]
    event_values = aligned_clinical["event"].astype(int)
    if not event_values.isin([0, 1]).all():
        raise ValueError("TCGA event labels must be binary (0 or 1).")
    total_events = int(event_values.sum())
    if expected_events is not None and total_events != expected_events:
        raise ValueError(
            f"aligned TCGA cohort has {total_events} events; "
            f"expected exactly {expected_events}."
        )

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        train_ids, test_ids = train_test_split(
            common_ids,
            train_size=n_train,
            test_size=n_test,
            stratify=event_values.to_numpy(),
            random_state=seed,
            shuffle=True,
        )
        train_ids = sorted(str(patient_id) for patient_id in train_ids)
        test_ids = sorted(str(patient_id) for patient_id in test_ids)
        manifest = {
            "version": 1,
            "dataset": "TCGA-BRCA",
            "seed": seed,
            "n_total": expected_total,
            "n_train": n_train,
            "n_test": n_test,
            "total_events": total_events,
            "train_events": int(aligned_clinical.loc[train_ids, "event"].sum()),
            "test_events": int(aligned_clinical.loc[test_ids, "event"].sum()),
            "train_ids": train_ids,
            "test_ids": test_ids,
        }
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        os.makedirs(manifest_dir, exist_ok=True)
        temporary_path = f"{manifest_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(temporary_path, manifest_path)

    required_metadata = {
        "version": 1,
        "dataset": "TCGA-BRCA",
        "seed": seed,
        "n_total": expected_total,
        "n_train": n_train,
        "n_test": n_test,
        "total_events": total_events,
    }
    for key, expected_value in required_metadata.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"TCGA split manifest has invalid {key}: "
                f"{manifest.get(key)!r}; expected {expected_value!r}."
            )

    train_ids = [str(patient_id) for patient_id in manifest.get("train_ids", [])]
    test_ids = [str(patient_id) for patient_id in manifest.get("test_ids", [])]
    if len(train_ids) != n_train or len(set(train_ids)) != n_train:
        raise ValueError("TCGA split manifest has invalid or duplicate train IDs.")
    if len(test_ids) != n_test or len(set(test_ids)) != n_test:
        raise ValueError("TCGA split manifest has invalid or duplicate test IDs.")
    if set(train_ids) & set(test_ids):
        raise ValueError("TCGA split manifest train and test IDs overlap.")
    if set(train_ids) | set(test_ids) != set(common_ids):
        raise ValueError(
            "TCGA split manifest IDs do not exactly cover the aligned cohort."
        )

    train_events = int(aligned_clinical.loc[train_ids, "event"].sum())
    test_events = int(aligned_clinical.loc[test_ids, "event"].sum())
    if manifest.get("train_events") != train_events:
        raise ValueError("TCGA split manifest train event count is inconsistent.")
    if manifest.get("test_events") != test_events:
        raise ValueError("TCGA split manifest test event count is inconsistent.")

    return TCGASplit(
        train_expression=aligned_expression.loc[train_ids],
        train_clinical=aligned_clinical.loc[train_ids],
        test_expression=aligned_expression.loc[test_ids],
        test_clinical=aligned_clinical.loc[test_ids],
        manifest=manifest,
    )


def build_propagated_features(
    propagator: DirectedPathwayPropagation,
    modulator: PersonalizedGraphModulator,
    A_enhanced: np.ndarray,
    dataset: PathwayDataset,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Materialize graph features in chunks while preserving sample order."""
    target_device = torch.device(device)
    propagator = propagator.to(target_device)

    if float(propagator.alpha.detach().cpu()) == 0.0:
        return dataset.X_tensor.to(target_device)

    feature_chunks = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch_idx = np.arange(start, stop)
            x_base_batch = dataset.X_tensor[batch_idx].to(target_device)
            mut_profiles = [
                dataset.mutation_dict.get(dataset.samples[i], [])
                for i in batch_idx
            ]
            A_batch = batch_build_patient_adjacencies(
                modulator, A_enhanced, mut_profiles
            ).to(target_device)
            feature_chunks.append(
                propagator.forward_batch_personalized(x_base_batch, A_batch)
            )

    return torch.cat(feature_chunks, dim=0)


def fit_graph_elastic_net_ensemble(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    genesets: Dict[str, List[str]],
    mutations: Dict[str, List[Tuple[str, str]]],
    modulator: PersonalizedGraphModulator,
    A_enhanced: np.ndarray,
    alpha_grid: List[float],
    penalizer_grid: List[float],
    l1_ratio_grid: List[float],
    n_folds: int,
    cv_repeats: int,
    min_selection_frequency: float,
    seed: int,
    batch_size: int,
    device: str,
) -> Dict:
    """Select propagation and fit a training-only Elastic-Net Cox ensemble."""
    dataset = PathwayDataset(
        expression,
        genesets,
        clinical,
        mutations,
        is_train=True,
        pathway_scoring_method="rank",
    )
    pathway_names = list(genesets)
    best_checkpoint = None

    for alpha in alpha_grid:
        propagator = DirectedPathwayPropagation(alpha=alpha)
        features_tensor = build_propagated_features(
            propagator,
            modulator,
            A_enhanced,
            dataset,
            batch_size=batch_size,
            device=device,
        )
        features = pd.DataFrame(
            features_tensor.detach().cpu().numpy(),
            index=dataset.samples,
            columns=pathway_names,
        )
        candidate = fit_elastic_net_cox_ensemble(
            features,
            dataset.clinical_df["time"],
            dataset.clinical_df["event"],
            penalizer_grid=penalizer_grid,
            l1_ratio_grid=l1_ratio_grid,
            n_splits=n_folds,
            n_repeats=cv_repeats,
            seed=seed,
            min_selection_frequency=min_selection_frequency,
        )
        candidate = dict(candidate)
        candidate["alpha"] = float(alpha)
        if (
            best_checkpoint is None
            or candidate["best_cv_c_index"]
            > best_checkpoint["best_cv_c_index"]
        ):
            best_checkpoint = candidate

    best_checkpoint["pathway_scaler"] = dataset.pathway_scaler.to_dict()
    best_features_tensor = build_propagated_features(
        DirectedPathwayPropagation(alpha=best_checkpoint["alpha"]),
        modulator,
        A_enhanced,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    best_features = pd.DataFrame(
        best_features_tensor.detach().cpu().numpy(),
        index=dataset.samples,
        columns=pathway_names,
    )
    train_risk = predict_cox_ensemble(best_features, best_checkpoint)
    best_checkpoint["train_c_index"] = float(
        concordance_index(
            dataset.times.numpy(),
            -train_risk,
            dataset.events.numpy(),
        )
    )
    return best_checkpoint


def summarize_personalization(
    sample_ids: List[str],
    mutations: Dict[str, List[Tuple[str, str]]],
    modulator: PersonalizedGraphModulator,
    A_enhanced: np.ndarray,
) -> Dict[str, int]:
    """Report whether mutation profiles and frozen rules affect the graph."""
    active_rule_edges = sum(
        A_enhanced[u, v] > 0 for u, v in modulator.rule_index
    )
    profiles_matched = 0
    patients_with_rules = 0
    patients_with_modulation = 0

    for sample_id in sample_ids:
        profile = mutations.get(sample_id, [])
        if profile:
            profiles_matched += 1
        if modulator.get_matched_rules_for_patient(profile):
            patients_with_rules += 1
        patient_adj = modulator.get_patient_adjacency(A_enhanced, profile)
        if np.any(np.abs(patient_adj - A_enhanced) > 1e-8):
            patients_with_modulation += 1

    return {
        "samples": len(sample_ids),
        "mutation_profiles_matched": profiles_matched,
        "patients_with_rule_matches": patients_with_rules,
        "patients_with_modulated_edges": patients_with_modulation,
        "active_rule_edges": int(active_rule_edges),
    }


def train_epoch(
    model: RidgeCoxSurvivalModel,
    propagator: DirectedPathwayPropagation,
    modulator: PersonalizedGraphModulator,
    A_enhanced: np.ndarray,
    dataset: PathwayDataset,
    batch_size: int,
    optimizer: optim.Optimizer
) -> float:
    """
    Training loop for one epoch (Sections 3.6 and 3.7 of the paper).

    For each batch:
    1. Dynamically synthesize patient-specific adjacency matrices A_i = A ⊙ Γ_i.
    2. Perform single-step directed graph pathway propagation x̃_i = x_i + α · x_i · A_i.
    3. Compute the Ridge-Cox negative log partial-likelihood loss.
    4. Update β through backpropagation.

    Args:
        model: RidgeCoxSurvivalModel instance.
        propagator: DirectedPathwayPropagation propagation operator.
        modulator: PersonalizedGraphModulator personalization modulator.
        A_enhanced: Enhanced population-level DAG adjacency matrix A (M x M).
        dataset: Training PathwayDataset.
        batch_size: Batch size.
        optimizer: PyTorch optimizer.

    Returns:
        avg_loss: Mean loss for this epoch.
    """
    model.train()
    device = next(model.parameters()).device
    # batch_size bounds graph-feature memory only. Cox partial likelihood is
    # computed once over the complete cohort so every event sees its full
    # risk set.
    x_tilde = build_propagated_features(
        propagator, modulator, A_enhanced, dataset, batch_size, str(device)
    )
    times = dataset.times.to(device)
    events = dataset.events.to(device)

    optimizer.zero_grad()
    pi = model(x_tilde)
    loss = model.compute_loss(pi, times, events)
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_cindex(
    model: RidgeCoxSurvivalModel,
    propagator: DirectedPathwayPropagation,
    modulator: PersonalizedGraphModulator,
    A_enhanced: np.ndarray,
    dataset: PathwayDataset
) -> float:
    """
    Compute the validation-set C-index (Section 4.2 of the paper).

    Uses lifelines concordance_index to compute Harrell's C-index.
    Higher PI indicates higher risk, so pass -pi to follow the lifelines convention
    (lifelines expects higher risk scores to correspond to shorter survival times).

    Args:
        model: Trained RidgeCoxSurvivalModel.
        propagator: Propagation operator.
        modulator: Personalization modulator.
        A_enhanced: Enhanced DAG adjacency matrix.
        dataset: Validation PathwayDataset.

    Returns:
        c_index: Harrell's concordance index.
    """
    model.eval()
    with torch.no_grad():
        x_base = dataset.X_tensor
        mut_profiles = [
            dataset.mutation_dict.get(s, []) for s in dataset.samples
        ]

        # Synthesize adjacency matrices in a batch and perform propagation.
        A_batch = batch_build_patient_adjacencies(modulator, A_enhanced, mut_profiles)
        x_tilde = propagator.forward_batch_personalized(x_base, A_batch)

        # Predict PI scores.
        pi = model(x_tilde).squeeze().cpu().numpy()
        times = dataset.times.numpy()
        events = dataset.events.numpy()

        # lifelines concordance_index expects higher risk scores → shorter event times.
        # Higher PI means higher risk, so negate it.
        c_index = concordance_index(times, -pi, events)

    return c_index


def run_5fold_cv_training(
    expr_path: str,
    clinical_path: str,
    genesets_path: str,
    reactome_path: str,
    rules_path: str,
    gene2pathway_path: str,
    mut_path: str,
    output_dir: str = "outputs/models",
    seed: int = 42,
    device: str = "cpu",
    alpha_grid: Optional[List[float]] = None,
    l2_reg_grid: Optional[List[float]] = None,
    model_type: str = "ridge_cox",
    penalizer_grid: Optional[List[float]] = None,
    l1_ratio_grid: Optional[List[float]] = None,
    n_folds: int = 5,
    cv_repeats: int = 3,
    min_selection_frequency: float = 0.6,
    learning_rate: float = 1e-3,
    max_epochs: int = 500,
    patience: int = 50,
    batch_size: int = 64,
    n_train: int = 694,
    n_test: int = 174,
    split_seed: int = 42,
    split_manifest_path: str = "data/processed/splits/tcga_brca_694_174_seed42.json",
    expected_events: Optional[int] = 102,
) -> Dict:
    """
    Main 5-fold cross-validation training workflow (Section 4.1 of the paper).

    On the TCGA-BRCA training set of 694 cases:
    1. Load raw data and prior knowledge (gene sets, Reactome hierarchy, and rule handbook).
    2. Build the population-level enhanced DAG.
    3. Grid-search for the optimal propagation coefficient α and L2 regularization parameter λ.
    4. Retrain on the full training set and save the final model.

    Args:
        expr_path: Path to the gene-expression-profile CSV.
        clinical_path: Path to the clinical-data CSV.
        genesets_path: Path to the JSON containing 331 pathway gene sets.
        reactome_path: Path to the Reactome-hierarchy JSON.
        rules_path: Path to global rule handbook R_1 JSON.
        gene2pathway_path: Path to the gene-to-pathway mapping JSON.
        mut_path: Path to the somatic mutation-signature JSON.
        output_dir: Model output directory.
        seed: Random seed.
        device: Compute device.

    Returns:
        results: Dictionary containing optimal hyperparameters, model path, and CV results.
    """
    # Set random seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MuDAG-Pro TCGA-BRCA 5-Fold CV Training (Section 4.1 of the paper)")
    print("=" * 60)

    # ================================================================
    # 1. Load metadata and prior knowledge.
    # ================================================================
    print("\n[1/6] Loading prior knowledge...")

    genesets = load_pathway_genesets(genesets_path)
    pathway_names = list(genesets.keys())
    M = len(pathway_names)
    print(f"  Loaded {M} pathway gene sets")

    parent_child_pairs = load_reactome_hierarchy(reactome_path)
    print(f"  Loaded {len(parent_child_pairs)} Reactome parent-child pairs")

    mutations = load_mutation_signatures(mut_path)
    print(f"  Loaded mutations for {len(mutations)} patients")

    with open(rules_path, "r", encoding="utf-8") as f:
        rules_handbook = json.load(f)
    print(f"  Loaded {len(rules_handbook)} rules from handbook")

    with open(gene2pathway_path, "r", encoding="utf-8") as f:
        gene2pathway = json.load(f)
    print(f"  Loaded gene-to-pathway mapping for {len(gene2pathway)} genes")

    # ================================================================
    # 2. Build population-level enhanced DAG A (Sections 3.2 and 3.4 of the paper).
    # ================================================================
    print("\n[2/6] Building enhanced DAG...")

    dag_builder = BaselineDAGBuilder(pathway_names)
    B, A_0 = dag_builder.build_baseline_dag(parent_child_pairs)
    print(f"  Baseline DAG: {A_0.shape}, {int(A_0.sum())} edges (transitive closure)")

    A_enhanced, W_new_dict = dag_builder.build_enhanced_dag(A_0, rules_handbook)
    print(f"  Enhanced DAG: {int(A_enhanced.sum())} edges "
          f"(+{len(W_new_dict)} new/modified cross-branch edges)")

    # ================================================================
    # 3. Read raw expression profiles and clinical data.
    # ================================================================
    print("\n[3/6] Loading expression & clinical data...")

    all_expression = load_patient_expression_data(expr_path)
    all_clinical = load_tcga_clinical_data(clinical_path, split=None)
    tcga_split = prepare_tcga_split(
        all_expression,
        all_clinical,
        manifest_path=split_manifest_path,
        n_train=n_train,
        n_test=n_test,
        seed=split_seed,
        expected_events=expected_events,
    )
    expr_df = tcga_split.train_expression
    clinical_df = tcga_split.train_clinical
    common_samples = expr_df.index

    print(
        f"  Aligned cohort: {tcga_split.manifest['n_total']} samples, "
        f"{tcga_split.manifest['total_events']} events"
    )
    print(
        f"  Fixed split: train={len(expr_df)} "
        f"(events={tcga_split.manifest['train_events']}), "
        f"test={len(tcga_split.test_expression)} "
        f"(events={tcga_split.manifest['test_events']})"
    )
    print(f"  Split manifest: {split_manifest_path}")

    # Detect whether precomputed pathway-score mode is in use.
    expr_cols = set(expr_df.columns)
    pathway_set = set(pathway_names)
    overlap = expr_cols & pathway_set
    precomputed = len(overlap) >= len(pathway_names) * 0.8

    if precomputed:
        print(f"  Mode: precomputed pathway scores ({len(overlap)}/{len(pathway_names)} pathways matched)")
    else:
        print(f"  Mode: gene expression → pathway score computation")

    alpha_grid = sorted({0.0, *(alpha_grid or [0.05, 0.10, 0.15, 0.20])})

    if model_type == "elastic_net_ensemble":
        print(
            "\n[4/6] Repeated stratified CV: alpha x Elastic-Net Cox..."
        )
        modulator = PersonalizedGraphModulator(
            pathway_names, rules_handbook, gene2pathway
        )
        personalization_summary = summarize_personalization(
            list(common_samples), mutations, modulator, A_enhanced
        )
        ensemble_checkpoint = fit_graph_elastic_net_ensemble(
            expression=expr_df,
            clinical=clinical_df,
            genesets=genesets,
            mutations=mutations,
            modulator=modulator,
            A_enhanced=A_enhanced,
            alpha_grid=alpha_grid,
            penalizer_grid=(
                penalizer_grid or [0.001, 0.01, 0.05, 0.1, 0.5]
            ),
            l1_ratio_grid=l1_ratio_grid or [0.1, 0.5, 0.9],
            n_folds=n_folds,
            cv_repeats=cv_repeats,
            min_selection_frequency=min_selection_frequency,
            seed=seed,
            batch_size=batch_size,
            device=device,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_checkpoint.update({
            "timestamp": timestamp,
            "n_features": M,
            "M": M,
            "personalization_summary": personalization_summary,
            "tcga_split": {
                key: value
                for key, value in tcga_split.manifest.items()
                if key not in {"train_ids", "test_ids"}
            } | {"manifest_path": os.path.abspath(split_manifest_path)},
            "training_config": {
                **ensemble_checkpoint.get("training_config", {}),
                "alpha_grid": alpha_grid,
                "feature_scoring": "within-sample-rank-pathway-v1",
                "feature_scaling": "tcga-training-median-iqr-v1",
                "feature_batch_size": batch_size,
            },
        })

        print("\n[5/6] Ensemble members fitted on repeated CV splits.")
        print("\n[6/6] Saving versioned ensemble checkpoint...")
        checkpoint_path = os.path.join(
            output_dir, f"mudag_pro_checkpoint_{timestamp}.pt"
        )
        best_path = os.path.join(output_dir, "best_model.pt")
        torch.save(ensemble_checkpoint, checkpoint_path)
        torch.save(ensemble_checkpoint, best_path)
        best_hyperparams = {
            "model_type": "elastic_net_ensemble",
            "alpha": ensemble_checkpoint["alpha"],
            "penalizer": ensemble_checkpoint["best_penalizer"],
            "l1_ratio": ensemble_checkpoint["best_l1_ratio"],
        }
        params_path = os.path.join(output_dir, "best_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_hyperparams, f, indent=4)
        print(
            f"  alpha*={ensemble_checkpoint['alpha']:.2f}, "
            f"penalizer*={ensemble_checkpoint['best_penalizer']:.3g}, "
            f"l1_ratio*={ensemble_checkpoint['best_l1_ratio']:.2f}"
        )
        print(
            f"  CV C-index={ensemble_checkpoint['best_cv_c_index']:.4f}, "
            f"train C-index={ensemble_checkpoint['train_c_index']:.4f}, "
            f"stable pathways={len(ensemble_checkpoint['stable_pathways'])}"
        )
        return {
            "best_alpha": ensemble_checkpoint["alpha"],
            "best_lambda": ensemble_checkpoint["best_penalizer"],
            "best_l1_ratio": ensemble_checkpoint["best_l1_ratio"],
            "best_cv_c_index": ensemble_checkpoint["best_cv_c_index"],
            "train_c_index": ensemble_checkpoint["train_c_index"],
            "model_path": best_path,
            "checkpoint_path": checkpoint_path,
            "params_path": params_path,
            "checkpoint": ensemble_checkpoint,
            "cv_results": ensemble_checkpoint["cv_results"],
        }
    if model_type != "ridge_cox":
        raise ValueError(f"Unsupported Cox model_type: {model_type}")

    # ================================================================
    # 4. Hyperparameter grid search (Sections 3.6, 3.7, and 4.1 of the paper).
    # ================================================================
    print("\n[4/6] Grid search: α (propagation) × λ (L2 regularization)...")

    # Grid-search ranges.
    l2_reg_grid = l2_reg_grid or [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    best_global_cindex = 0.0
    best_hyperparams = {"alpha": 0.10, "l2_reg": 1e-3}
    all_cv_results = []

    cv_splits = build_cv_splits(
        clinical_df["event"].to_numpy(), n_folds=n_folds, seed=seed
    )

    fold_datasets = []
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        train_expr = expr_df.iloc[train_idx]
        val_expr = expr_df.iloc[val_idx]
        train_clin = clinical_df.iloc[train_idx]
        val_clin = clinical_df.iloc[val_idx]

        train_ds = PathwayDataset(
            train_expr, genesets, train_clin, mutations,
            is_train=True,
            precomputed_pathways=precomputed,
        )
        val_ds = PathwayDataset(
            val_expr, genesets, val_clin, mutations,
            is_train=False,
            train_gene_means=(
                train_ds.gene_means if not precomputed else None
            ),
            train_gene_stds=(
                train_ds.gene_stds if not precomputed else None
            ),
            precomputed_pathways=precomputed,
        )
        fold_datasets.append((train_ds, val_ds))
        print(
            f"  Fold {fold + 1}: train events="
            f"{int(train_ds.events.sum())}, validation events="
            f"{int(val_ds.events.sum())}"
        )

    modulator = PersonalizedGraphModulator(
        pathway_names, rules_handbook, gene2pathway
    )
    personalization_summary = summarize_personalization(
        list(common_samples), mutations, modulator, A_enhanced
    )
    print(
        "  Personalization: "
        f"mutation profiles {personalization_summary['mutation_profiles_matched']}"
        f"/{personalization_summary['samples']}, rule matches "
        f"{personalization_summary['patients_with_rule_matches']}, "
        f"modulated patients "
        f"{personalization_summary['patients_with_modulated_edges']}, "
        f"active rule edges {personalization_summary['active_rule_edges']}"
    )
    if (
        personalization_summary["mutation_profiles_matched"] > 0
        and personalization_summary["patients_with_modulated_edges"] == 0
    ):
        print(
            "  [Warning] Mutation profiles were loaded but no patient graph "
            "was changed; CV may select alpha=0."
        )

    for alpha in alpha_grid:
        propagator = DirectedPathwayPropagation(alpha=alpha)
        fold_features = []
        for train_ds, val_ds in fold_datasets:
            train_features = build_propagated_features(
                propagator, modulator, A_enhanced, train_ds,
                batch_size, device,
            )
            val_features = build_propagated_features(
                propagator, modulator, A_enhanced, val_ds,
                batch_size, device,
            )
            fold_features.append((train_features, val_features))

        for l2_reg in l2_reg_grid:
            fold_cindices = []
            fold_epochs = []

            for fold, ((train_ds, val_ds), features) in enumerate(
                zip(fold_datasets, fold_features)
            ):
                train_features, val_features = features
                model = RidgeCoxSurvivalModel(
                    in_features=M, l2_reg=l2_reg
                ).to(device)
                history = train_ridge_cox(
                    model,
                    train_features,
                    train_ds.times.to(device),
                    train_ds.events.to(device),
                    val_features,
                    val_ds.times.to(device),
                    val_ds.events.to(device),
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    patience=patience,
                    verbose=False,
                )
                fold_epochs.append(len(history["train_loss"]))

                model.eval()
                with torch.no_grad():
                    pi = model(val_features).squeeze().cpu().numpy()
                c_idx = concordance_index(
                    val_ds.times.numpy(), -pi, val_ds.events.numpy()
                )
                fold_cindices.append(c_idx)

            mean_cindex = np.mean(fold_cindices)
            std_cindex = np.std(fold_cindices)

            result_entry = {
                "alpha": alpha,
                "l2_reg": l2_reg,
                "mean_c_index": float(mean_cindex),
                "std_c_index": float(std_cindex),
                "fold_c_indices": [float(c) for c in fold_cindices],
                "fold_epochs": fold_epochs,
            }
            all_cv_results.append(result_entry)

            print(f"  α={alpha:.2f}, λ={l2_reg:.0e}: "
                  f"C-index = {mean_cindex:.4f} ± {std_cindex:.4f}")

            if mean_cindex > best_global_cindex:
                best_global_cindex = mean_cindex
                best_hyperparams = {"alpha": alpha, "l2_reg": l2_reg}

    print(f"\n  [Optimal] α* = {best_hyperparams['alpha']:.2f}, "
          f"λ* = {best_hyperparams['l2_reg']:.0e}, "
          f"CV C-index = {best_global_cindex:.4f}")

    final_epochs = select_final_training_epochs(
        all_cv_results, best_hyperparams, max_epochs
    )
    print(
        f"  Final training epochs = {final_epochs} "
        "(median early-stop epoch of the best CV setting)"
    )

    # ================================================================
    # 5. Retrain the final model on the full training set (Section 4.1 of the paper).
    # ================================================================
    print(f"\n[5/6] Retraining final model on full training set...")

    full_train_ds = PathwayDataset(
        expr_df, genesets, clinical_df, mutations,
        is_train=True,
        precomputed_pathways=precomputed,
    )

    final_propagator = DirectedPathwayPropagation(
        alpha=best_hyperparams["alpha"]
    )
    final_model = RidgeCoxSurvivalModel(
        in_features=M, l2_reg=best_hyperparams["l2_reg"]
    ).to(device)

    full_train_features = build_propagated_features(
        final_propagator, modulator, A_enhanced, full_train_ds,
        batch_size, device,
    )
    final_history = train_ridge_cox(
        final_model,
        full_train_features,
        full_train_ds.times.to(device),
        full_train_ds.events.to(device),
        learning_rate=learning_rate,
        max_epochs=final_epochs,
        patience=final_epochs + 1,
        verbose=True,
    )

    # Compute the training-set C-index.
    final_model.eval()
    with torch.no_grad():
        train_pi = final_model(full_train_features).squeeze().cpu().numpy()
    train_c_index = concordance_index(
        full_train_ds.times.numpy(), -train_pi, full_train_ds.events.numpy()
    )
    print(f"  Training C-index: {train_c_index:.4f}")

    # ================================================================
    # 6. Save the model and hyperparameters (Section 4.1 of the paper).
    # ================================================================
    print("\n[6/6] Saving model checkpoint...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model weights.
    model_path = os.path.join(output_dir, "mudag_pro_model.pth")
    torch.save(final_model.state_dict(), model_path)

    # Save the full checkpoint (including weights, hyperparameters, and CV results).
    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "beta": final_model.linear.weight.detach().cpu().numpy().squeeze(),
        "alpha": best_hyperparams["alpha"],
        "best_lambda": best_hyperparams["l2_reg"],
        "l2_reg": best_hyperparams["l2_reg"],
        "best_cv_c_index": best_global_cindex,
        "train_c_index": train_c_index,
        "cv_results": all_cv_results,
        "n_features": M,
        "pathway_names": pathway_names,
        "timestamp": timestamp,
        "M": M,
        "personalization_summary": personalization_summary,
        "tcga_split": {
            key: value
            for key, value in tcga_split.manifest.items()
            if key not in {"train_ids", "test_ids"}
        } | {"manifest_path": os.path.abspath(split_manifest_path)},
        "training_config": {
            "alpha_grid": alpha_grid,
            "l2_reg_grid": l2_reg_grid,
            "n_folds": n_folds,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "patience": patience,
            "feature_batch_size": batch_size,
            "selected_final_epochs": final_epochs,
            "final_epochs": len(final_history["train_loss"]),
        },
    }

    checkpoint_path = os.path.join(
        output_dir, f"mudag_pro_checkpoint_{timestamp}.pt"
    )
    torch.save(checkpoint, checkpoint_path)

    # Also save as best_model.pt for convenient loading.
    best_path = os.path.join(output_dir, "best_model.pt")
    torch.save(checkpoint, best_path)

    # Save the hyperparameters JSON.
    params_path = os.path.join(output_dir, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_hyperparams, f, indent=4)

    print(f"  Model weights: {model_path}")
    print(f"  Full checkpoint: {checkpoint_path}")
    print(f"  Best params: {params_path}")
    print(f"  α* = {best_hyperparams['alpha']:.2f}, "
          f"λ* = {best_hyperparams['l2_reg']:.0e}")
    print(f"  CV C-index = {best_global_cindex:.4f}")
    print(f"  Train C-index = {train_c_index:.4f}")

    return {
        "best_alpha": best_hyperparams["alpha"],
        "best_lambda": best_hyperparams["l2_reg"],
        "best_cv_c_index": best_global_cindex,
        "train_c_index": train_c_index,
        "model_path": model_path,
        "checkpoint_path": checkpoint_path,
        "params_path": params_path,
        "checkpoint": checkpoint,
        "cv_results": all_cv_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MuDAG-Pro TCGA-BRCA 5-Fold CV Training"
    )
    parser.add_argument(
        "--expr", type=str,
        default="data/raw/tcga_brca/expr_train.csv",
        help="Gene expression profile CSV path",
    )
    parser.add_argument(
        "--clinical", type=str,
        default="data/raw/tcga_brca/clinical_train.csv",
        help="Clinical data CSV path",
    )
    parser.add_argument(
        "--genesets", type=str,
        default="data/processed/pathway_genesets/331_pathways.json",
        help="331 pathway gene sets JSON path",
    )
    parser.add_argument(
        "--reactome", type=str,
        default="data/raw/reactome/hierarchy.json",
        help="Reactome hierarchy JSON path",
    )
    parser.add_argument(
        "--rules", type=str,
        default="knowledge_engine/rules_handbook.json",
        help="Rule handbook R_1 JSON path",
    )
    parser.add_argument(
        "--gene2pathway", type=str,
        default="data/processed/gene2pathway.json",
        help="Gene-to-pathway mapping JSON path",
    )
    parser.add_argument(
        "--mutations", type=str,
        default="data/raw/tcga_brca/mutations.json",
        help="Somatic mutation signatures JSON path",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/models",
        help="Model output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    run_5fold_cv_training(
        expr_path=args.expr,
        clinical_path=args.clinical,
        genesets_path=args.genesets,
        reactome_path=args.reactome,
        rules_path=args.rules,
        gene2pathway_path=args.gene2pathway,
        mut_path=args.mutations,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
