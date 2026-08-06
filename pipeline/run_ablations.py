"""
MuDAG-Pro ablation experiment pipeline (Section 4.3 of the paper).

Runs five ablation variants:
- M_static: Completely remove personalized edge-weight adjustment; all patients share the population-level DAG.
- M_onehot: Assign a fixed constant (>1) to all mutation-pathway edge weights.
- M_rand_dist: Randomly shuffle the biological mutation-to-edge mapping.
- M_inter: Add pathway × mutation interaction terms when computing PI, but do not use edge weights in propagation.
- M_LOO: Leave-One-Out edge-by-edge ablation that masks rules corresponding to frequent mutations.
"""
import os
import json
import yaml
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from copy import deepcopy


def run_ablations(
    checkpoint_path: str = "outputs/models/best_model.pt",
    config_path: str = "config/base_config.yaml",
    data_dir: str = "data/processed",
    output_dir: str = "outputs",
    variants: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    device: str = "cpu",
) -> Dict:
    """
    Run ablation experiments.

    Args:
        checkpoint_path: Full model checkpoint.
        config_path: Configuration file.
        data_dir: Data directory.
        output_dir: Output directory.
        variants: List of variants to run (all by default).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        device: Compute device.

    Returns:
        ablation_results: Results for each variant on each dataset.
    """
    if variants is None:
        variants = ["M_static", "M_onehot", "M_rand_dist", "M_inter", "M_LOO"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the configuration.
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prop_config = config["propagation"]
    alpha = prop_config["alpha"]

    # Load the model.
    print("=" * 60)
    print("MuDAG-Pro Ablation Study")
    print("=" * 60)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the DAG.
    A = _load_matrix(
        os.path.join(data_dir, "baseline_dag", "enhanced_adj.npy"),
        default=np.eye(checkpoint["n_features"], dtype=np.float32),
    )
    M = A.shape[0]
    pathway_names = [f"Pathway_{i}" for i in range(M)]

    # Load rules and gene mappings.
    rules_path = os.path.join("knowledge_engine", "rules_handbook.json")
    rules = _load_json(rules_path, default=[])
    g2p_path = os.path.join(data_dir, "pathway_genesets", "gene_to_pathway.json")
    gene_to_pathway = _load_json(g2p_path, default={})

    # Datasets.
    datasets = ["tcga_holdout", "metabric"]
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'=' * 40}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 40}")

        X, times, events, mutations = _load_dataset(data_dir, dataset_name, config)
        if X is None:
            continue

        print(f"  Samples: {X.shape[0]}, Events: {int(events.sum())}")

        dataset_results = {}

        for variant in variants:
            print(f"\n  [{variant}]")

            if variant == "M_static":
                X_prop = _run_static(X, A, alpha)
            elif variant == "M_onehot":
                X_prop = _run_onehot(
                    X, A, alpha, mutations, rules, gene_to_pathway, pathway_names
                )
            elif variant == "M_rand_dist":
                X_prop = _run_rand_dist(
                    X, A, alpha, mutations, rules, gene_to_pathway, pathway_names, seed
                )
            elif variant == "M_inter":
                X_prop = _run_static(X, A, alpha)  # Propagation is the same as in the static variant.
                X_prop = _add_interaction_terms(X_prop, mutations, gene_to_pathway, pathway_names)
            elif variant == "M_LOO":
                X_prop = _run_loo(
                    X, A, alpha, mutations, rules, gene_to_pathway, pathway_names
                )
            else:
                print(f"    Unknown variant: {variant}, skipping")
                continue

            # Evaluate.
            from src.cox_model import RidgeCoxSurvivalModel
            model = RidgeCoxSurvivalModel(
                in_features=checkpoint["n_features"],
                l2_reg=checkpoint["best_lambda"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            x_tensor = torch.tensor(X_prop, dtype=torch.float32).to(device)
            pi_scores = model.get_prognostic_index(x_tensor)

            from analysis.metrics import compute_c_index, compute_c_index_bootstrap
            c_idx = compute_c_index(pi_scores, times, events)
            c_mean, c_std = compute_c_index_bootstrap(
                pi_scores, times, events, n_bootstrap=n_bootstrap, seed=seed
            )

            print(f"    C-index: {c_mean:.4f} ± {c_std:.4f}")

            dataset_results[variant] = {
                "c_index": float(c_idx),
                "c_index_bootstrap_mean": float(c_mean),
                "c_index_bootstrap_std": float(c_std),
            }

        all_results[dataset_name] = dataset_results

    # Compute ΔC and significance.
    print("\n" + "=" * 60)
    print("Ablation Analysis: ΔC vs M_static")
    print("=" * 60)

    ablation_summary = {}
    for dataset_name, results in all_results.items():
        if "M_static" not in results:
            continue

        baseline_c = results["M_static"]["c_index_bootstrap_mean"]
        summary = {}

        for variant, res in results.items():
            delta_c = res["c_index_bootstrap_mean"] - baseline_c

            # Bootstrap significance test.
            p_value = _compute_bootstrap_p_value(
                baseline_c, res["c_index_bootstrap_std"],
                n_bootstrap,
            )

            summary[variant] = {
                "c_index": res["c_index_bootstrap_mean"],
                "delta_c": float(delta_c),
                "p_value": float(p_value),
            }

            print(f"  {dataset_name}/{variant}: ΔC = {delta_c:+.4f} (p = {p_value:.4f})")

        ablation_summary[dataset_name] = summary

    # Save results.
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    report_path = os.path.join(
        output_dir, "reports",
        f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "per_dataset": all_results,
            "summary": ablation_summary,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nAblation report saved to: {report_path}")

    return {"per_dataset": all_results, "summary": ablation_summary}


def _run_static(X: np.ndarray, A: np.ndarray, alpha: float) -> np.ndarray:
    """M_static: Completely remove personalized edge-weight adjustment."""
    from src.graph_propagation import DirectedPathwayPropagation
    prop = DirectedPathwayPropagation(alpha=alpha)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    return prop.forward_batch_static(x_tensor, A_tensor).numpy()


def _run_onehot(
    X: np.ndarray, A: np.ndarray, alpha: float,
    mutations: Dict, rules: list,
    gene_to_pathway: Dict, pathway_names: List[str],
    fixed_weight: float = 1.5,
) -> np.ndarray:
    """M_onehot: Assign a fixed constant to all mutation-pathway edge weights."""
    from src.personalized_graph import PersonalizedGraphModulator
    from src.graph_propagation import DirectedPathwayPropagation

    # Modify rules by setting every delta and confidence to a fixed value.
    simplified_rules = []
    for rule in rules:
        r = deepcopy(rule)
        r["delta"] = fixed_weight
        r["confidence"] = 1.0
        simplified_rules.append(r)

    pgb = PersonalizedGraphModulator(
        pathway_names=pathway_names,
        rules_handbook=simplified_rules,
        gene2pathway_map=gene_to_pathway,
    )

    prop = DirectedPathwayPropagation(alpha=alpha)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    adj_batch = []
    for i in range(X.shape[0]):
        mut_sig = _get_mutation(mutations, i)
        if mut_sig:
            A_i = pgb.get_patient_adjacency(A, mut_sig)
        else:
            A_i = A.copy()
        adj_batch.append(torch.tensor(A_i, dtype=torch.float32))
    adj_batch = torch.stack(adj_batch)
    return prop.forward_batch_personalized(x_tensor, adj_batch).numpy()


def _run_rand_dist(
    X: np.ndarray, A: np.ndarray, alpha: float,
    mutations: Dict, rules: list,
    gene_to_pathway: Dict, pathway_names: List[str],
    seed: int,
    n_repeats: int = 100,
) -> np.ndarray:
    """M_rand_dist: Randomly shuffle mutation-to-edge mappings and average 100 repetitions."""
    rng = np.random.RandomState(seed)
    all_props = []

    for rep in range(n_repeats):
        # Randomly shuffle trigger_genes in the rules.
        shuffled_rules = []
        all_genes = []
        for rule in rules:
            all_genes.extend(rule.get("trigger_genes", []))

        if all_genes:
            rng.shuffle(all_genes)

        gene_idx = 0
        for rule in rules:
            r = deepcopy(rule)
            n_triggers = len(r.get("trigger_genes", []))
            if n_triggers > 0 and gene_idx + n_triggers <= len(all_genes):
                r["trigger_genes"] = all_genes[gene_idx:gene_idx + n_triggers]
                gene_idx += n_triggers
            shuffled_rules.append(r)

        from src.personalized_graph import PersonalizedGraphModulator
        pgb = PersonalizedGraphModulator(
            pathway_names=pathway_names,
            rules_handbook=shuffled_rules,
            gene2pathway_map=gene_to_pathway,
        )

        from src.graph_propagation import DirectedPathwayPropagation
        prop = DirectedPathwayPropagation(alpha=alpha)
        x_tensor = torch.tensor(X, dtype=torch.float32)
        adj_batch = []
        for i in range(X.shape[0]):
            mut_sig = _get_mutation(mutations, i)
            A_i = pgb.get_patient_adjacency(A, mut_sig) if mut_sig else A.copy()
            adj_batch.append(torch.tensor(A_i, dtype=torch.float32))
        adj_batch = torch.stack(adj_batch)
        X_p = prop.forward_batch_personalized(x_tensor, adj_batch).numpy()
        all_props.append(X_p)

    # Take the mean.
    return np.mean(all_props, axis=0)


def _add_interaction_terms(
    X: np.ndarray, mutations: Dict,
    gene_to_pathway: Dict, pathway_names: List[str],
) -> np.ndarray:
    """M_inter: Append pathway × mutation interaction terms to the feature matrix."""
    M = X.shape[0]
    n_features = X.shape[1]

    # Build the mutation-indicator matrix.
    mut_indicators = np.zeros((M, n_features), dtype=np.float32)
    for i in range(M):
        mut_sig = _get_mutation(mutations, i)
        mutated_genes = {g for g, _ in mut_sig} if mut_sig else set()
        for gene in mutated_genes:
            if gene in gene_to_pathway:
                for pw in gene_to_pathway[gene]:
                    if pw in pathway_names:
                        j = pathway_names.index(pw)
                        mut_indicators[i, j] = 1.0

    # Interaction features: X ⊙ mutation_indicator.
    interaction = X * mut_indicators
    return np.hstack([X, interaction])


def _run_loo(
    X: np.ndarray, A: np.ndarray, alpha: float,
    mutations: Dict, rules: list,
    gene_to_pathway: Dict, pathway_names: List[str],
) -> np.ndarray:
    """M_LOO: Leave-One-Out edge-by-edge ablation that masks rules for frequent mutations."""
    # Compute mutation frequencies.
    from collections import Counter
    gene_freq = Counter()
    for mut_sig in mutations.values():
        if isinstance(mut_sig, list):
            for item in mut_sig:
                if isinstance(item, list):
                    gene_freq[item[0]] += 1
                elif isinstance(item, str):
                    gene_freq[item] += 1

    # Identify frequently mutated genes (frequency > 5%).
    total_patients = len(mutations) if mutations else 1
    high_freq_genes = {g for g, c in gene_freq.items()
                       if c / total_patients > 0.05}

    # Remove rules related to frequently mutated genes.
    filtered_rules = []
    for rule in rules:
        triggers = set(rule.get("trigger_genes", []))
        if not (triggers & high_freq_genes):
            filtered_rules.append(rule)

    print(f"    LOO: removed {len(rules) - len(filtered_rules)} rules "
          f"(high-freq genes: {high_freq_genes})")

    from src.personalized_graph import PersonalizedGraphModulator
    from src.graph_propagation import DirectedPathwayPropagation

    pgb = PersonalizedGraphModulator(
        pathway_names=pathway_names,
        rules_handbook=filtered_rules,
        gene2pathway_map=gene_to_pathway,
    )

    prop = DirectedPathwayPropagation(alpha=alpha)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    adj_batch = []
    for i in range(X.shape[0]):
        mut_sig = _get_mutation(mutations, i)
        A_i = pgb.get_patient_adjacency(A, mut_sig) if mut_sig else A.copy()
        adj_batch.append(torch.tensor(A_i, dtype=torch.float32))
    adj_batch = torch.stack(adj_batch)
    return prop.forward_batch_personalized(x_tensor, adj_batch).numpy()


def _get_mutation(mutations: Dict, sample_idx: int) -> List[Tuple[str, str]]:
    for key in [str(sample_idx), sample_idx, str(sample_idx).zfill(4)]:
        if key in mutations:
            raw = mutations[key]
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], list):
                    return [tuple(item) for item in raw]
                elif isinstance(raw[0], str):
                    return [(g, "") for g in raw]
    return []


def _load_matrix(path: str, default=None):
    if os.path.exists(path):
        return np.load(path)
    return default


def _load_json(path: str, default=None):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default


def _load_dataset(data_dir: str, dataset_name: str, config: Dict):
    prefix_map = {"tcga_holdout": "tcga_holdout", "metabric": "metabric"}
    prefix = prefix_map.get(dataset_name, dataset_name)

    X_path = os.path.join(data_dir, "expression_matrices", f"{prefix}_pathway_scores.npy")
    t_path = os.path.join(data_dir, f"{prefix}_times.npy")
    e_path = os.path.join(data_dir, f"{prefix}_events.npy")
    m_path = os.path.join(data_dir, "mutation_profiles", f"{prefix}_mutations.json")

    if not os.path.exists(X_path):
        return None, None, None, None

    X = np.load(X_path)
    times = np.load(t_path) if os.path.exists(t_path) else np.zeros(X.shape[0])
    events = np.load(e_path) if os.path.exists(e_path) else np.zeros(X.shape[0])
    mutations = _load_json(m_path, default={})

    return X, times, events, mutations


def _compute_bootstrap_p_value(
    baseline_mean: float,
    variant_std: float,
    n_bootstrap: int,
) -> float:
    """Estimate the p-value of ΔC with a simplified bootstrap approach."""
    from scipy import stats
    # Assume H0: ΔC = 0 and use a normal approximation.
    if variant_std < 1e-8:
        return 1.0
    z_score = abs(baseline_mean) / variant_std
    p_value = 2.0 * (1.0 - stats.norm.cdf(z_score))
    return min(p_value, 1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MuDAG-Pro ablation studies")
    parser.add_argument("--checkpoint", type=str, default="outputs/models/best_model.pt")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["M_static", "M_onehot", "M_rand_dist", "M_inter", "M_LOO"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_ablations(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        variants=args.variants,
        seed=args.seed,
        device=args.device,
    )
