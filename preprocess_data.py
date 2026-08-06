"""
MuDAG-Pro data-preprocessing pipeline (331-pathway version).

Reads raw data from the datasets_csv and Data directories and converts it to the standard format expected by the pipeline.

Data sources:
- datasets_csv/metadata/combine_signatures.csv: Gene sets for 331 pathways (200 genes × 331 pathways).
- datasets_csv/metadata/hallmarks_signatures.csv: Gene sets for 50 Hallmark pathways.
- datasets_csv/raw_rna_data/combine/brca/rna_clean.csv: TCGA-BRCA expression profiles (940 × 5000).
- datasets_csv/metadata/tcga_brca.csv: TCGA-BRCA clinical data (including train/test split).
- Data/reactome_data/ReactomePathwaysRelation.txt: Reactome hierarchy.
- Data/reactome_data/ReactomePathways.txt: Reactome pathway-name mapping.

Outputs (data/processed/):
- pathway_genesets/pathway_genesets.json: Gene sets for 331 pathways.
- pathway_genesets/gene_to_pathway.json: Gene-to-pathway mapping.
- pathway_genesets/reactome_hierarchy.json: Reactome hierarchical edges.
- expression_matrices/tcga_pathway_scores.npy: TCGA pathway scores (N×331).
- tcga_times.npy, tcga_events.npy: Survival data.
- tcga_train_idx.npy, tcga_test_idx.npy: Train/test split.
- baseline_dag/baseline_adj.npy: Baseline DAG adjacency matrix.
- baseline_dag/enhanced_adj.npy: Enhanced DAG.
- mutation_profiles/tcga_mutations.json: Mutation signatures, when available.
"""
import os
import sys
import json
import gzip
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


# ================================================================
# Path configuration.
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets_csv")
METADATA_DIR = os.path.join(DATASETS_DIR, "metadata")
RNA_DIR = os.path.join(DATASETS_DIR, "raw_rna_data", "combine")
REACTOME_DIR = os.path.join(BASE_DIR, "Data", "reactome_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MUTATION_DIR = os.path.join(BASE_DIR, "Data", "TCGA-BRCA")


def ensure_dirs():
    """Ensure that all output directories exist."""
    dirs = [
        os.path.join(PROCESSED_DIR, "expression_matrices"),
        os.path.join(PROCESSED_DIR, "baseline_dag"),
        os.path.join(PROCESSED_DIR, "mutation_profiles"),
        os.path.join(PROCESSED_DIR, "pathway_genesets"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_pathway_genesets() -> Dict[str, List[str]]:
    """
    Load 331 pathway gene sets from combine_signatures.csv.

    File format: each column is a pathway and rows contain gene names (200 rows × 331 columns).
    The first row contains pathway names; subsequent rows contain genes in each pathway.
    """
    sig_path = os.path.join(METADATA_DIR, "combine_signatures.csv")
    print(f"[1/7] Loading pathway gene sets from combine_signatures.csv...")

    df = pd.read_csv(sig_path)
    genesets = {}
    for pathway in df.columns:
        pathway = pathway.strip()
        genes = df[pathway].dropna().tolist()
        genes = [str(g).strip() for g in genes if str(g).strip()]
        genesets[pathway] = genes

    pathway_names = list(genesets.keys())
    n_pathways = len(pathway_names)
    n_genes_all = sum(len(v) for v in genesets.values())
    print(f"  Loaded {n_pathways} pathways, {n_genes_all} gene-pathway associations")

    return genesets


def compute_pathway_scores(
    expr_df: pd.DataFrame,
    genesets: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compute pathway activation scores (Section 3.1 of the paper).

    For each pathway: s_k = (1/|G_k|) · Σ_{g∈G_k} z_g,
    where z_g is the Z-score-normalized gene-expression value.

    The RNA data is already Z-score normalized, so compute the mean directly.
    """
    print(f"\n[2/7] Computing pathway activation scores...")

    pathway_names = list(genesets.keys())
    scores = {}

    expr_genes = set(expr_df.columns)
    matched_count = 0
    missing_count = 0

    for p_name in pathway_names:
        genes_in_pw = genesets[p_name]
        available = [g for g in genes_in_pw if g in expr_genes]

        if len(available) > 0:
            pathway_score = expr_df[available].mean(axis=1)
            matched_count += 1
        else:
            pathway_score = pd.Series(0.0, index=expr_df.index)
            missing_count += 1

        scores[p_name] = pathway_score

    print(f"  Matched: {matched_count} pathways, Missing genes: {missing_count} pathways")
    print(f"  Avg genes per pathway: {sum(len(v) for v in genesets.values()) / len(genesets):.1f}")

    return pd.DataFrame(scores, index=expr_df.index)


def load_clinical_data() -> pd.DataFrame:
    """
    Load TCGA-BRCA clinical data.

    tcga_brca.csv columns:
    - case_id: Patient ID.
    - survival_months: Survival time (months).
    - censorship: 0=event (death), 1=censored (alive)  ← opposite to the paper.
    - train: 1.0=training set, 0.0=test set.
    """
    print(f"\n[3/7] Loading clinical data...")

    clinical_path = os.path.join(METADATA_DIR, "tcga_brca.csv")
    df = pd.read_csv(clinical_path)

    # Deduplicate (each case_id may have multiple slides).
    df_unique = df.drop_duplicates(subset="case_id").copy()
    df_unique = df_unique.set_index("case_id")

    # Convert to the format used in the paper.
    # Paper: event=1 means death/recurrence; event=0 means censored.
    # Data: censorship=0 means death; censorship=1 means censored.
    df_unique["event"] = 1.0 - df_unique["censorship"].astype(float)

    # Convert to days (the paper uses days).
    df_unique["time"] = df_unique["survival_months"].astype(float) * 30.44

    # Train/test split.
    df_unique["is_train"] = df_unique["train"].astype(float)

    n_train = int(df_unique["is_train"].sum())
    n_test = len(df_unique) - n_train
    n_events = int(df_unique["event"].sum())
    n_censored = len(df_unique) - n_events

    print(f"  Total: {len(df_unique)} patients")
    print(f"  Train: {n_train}, Test: {n_test}")
    print(f"  Events: {n_events} (deceased), Censored: {n_censored} (alive)")

    return df_unique


def build_dag_from_reactome(
    pathway_names: List[str],
    genesets: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Build a DAG from Reactome hierarchical relationships (Section 3.2 of the paper).

    Strategy: Because Reactome pathway names (such as "R-BTA-73843") do not directly correspond
    to our pathway names (such as "ADP_signalling_through_P2Y_purinoceptor_12"),
    use the following methods:
    1. Infer parent-child relationships between pathways from gene overlap.
    2. Use direct Reactome mappings when available.

    For the 331 pathways, build hierarchical relationships using the gene-overlap coefficient:
    if more than 80% of pathway A's gene set is contained in pathway B, then A is a child pathway of B.
    """
    print(f"\n[4/7] Building DAG from pathway gene overlap...")

    M = len(pathway_names)
    overlap_threshold = 0.30  # Gene-overlap threshold.

    parent_child_pairs = []

    # Compute gene overlap for all pathway pairs.
    for i, pw_i in enumerate(pathway_names):
        genes_i = set(genesets[pw_i])
        for j, pw_j in enumerate(pathway_names):
            if i == j:
                continue
            genes_j = set(genesets[pw_j])
            if len(genes_i) == 0 or len(genes_j) == 0:
                continue

            overlap = genes_i & genes_j
            # If most genes in i are contained in j, then i may be a parent node of j.
            overlap_ratio = len(overlap) / len(genes_i)

            if overlap_ratio >= overlap_threshold:
                parent_child_pairs.append((pw_i, pw_j))

    print(f"  Generated {len(parent_child_pairs)} parent-child pairs "
          f"(overlap ≥ {overlap_threshold:.0%})")

    # Build the DAG with BaselineDAGBuilder.
    from src.baseline_dag_builder import BaselineDAGBuilder

    builder = BaselineDAGBuilder(pathway_names)
    B, A_0 = builder.build_baseline_dag(parent_child_pairs)

    print(f"  Direct edges: {int(B.sum())}")
    print(f"  Transitive closure edges: {int(A_0.sum())}")

    return A_0, parent_child_pairs


def build_reactome_hierarchy_json(
    parent_child_pairs: List[Tuple[str, str]],
):
    """Save Reactome hierarchical relationships as JSON."""
    output_path = os.path.join(
        PROCESSED_DIR, "pathway_genesets", "reactome_hierarchy.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parent_child_pairs, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(parent_child_pairs)} hierarchy edges to: {output_path}")


def load_mutation_data() -> Dict[str, List[Tuple[str, str]]]:
    """
    Load TCGA-BRCA somatic mutation data.

    Read from aligned_mutation_X.tsv with the format:
    sample,gene,chrom,start,end,ref,alt,effect,...
    """
    print(f"\n[5/7] Loading mutation data...")

    mut_path = os.path.join(MUTATION_DIR, "aligned_mutation_X.tsv")
    if not os.path.exists(mut_path):
        print(f"  Mutation file not found: {mut_path}")
        return {}

    mut_df = pd.read_csv(mut_path, sep='\t')
    print(f"  Raw mutations: {len(mut_df)} records")

    # Build patient mutation signatures.
    mutations = defaultdict(list)
    for _, row in mut_df.iterrows():
        sample = str(row["sample"])
        gene = str(row["gene"])
        effect = str(row.get("effect", ""))
        mutations[sample].append([gene, effect])

    mut_dict = dict(mutations)
    print(f"  Mutation signatures: {len(mut_dict)} patients")

    # Save.
    output_path = os.path.join(
        PROCESSED_DIR, "mutation_profiles", "tcga_mutations.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mut_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {output_path}")

    return mut_dict


def save_processed_data(
    genesets: Dict[str, List[str]],
    pathway_scores: pd.DataFrame,
    clinical_df: pd.DataFrame,
    A_0: np.ndarray,
    A_enhanced: np.ndarray,
    parent_child_pairs: List[Tuple[str, str]],
):
    """Save all preprocessed data to data/processed/."""
    print(f"\n[6/7] Saving processed data...")

    pathway_names = list(genesets.keys())

    # 1. Pathway gene sets.
    geneset_path = os.path.join(
        PROCESSED_DIR, "pathway_genesets", "pathway_genesets.json"
    )
    with open(geneset_path, 'w', encoding='utf-8') as f:
        json.dump(genesets, f, indent=2, ensure_ascii=False)
    print(f"  Pathway genesets: {geneset_path}")

    # 2. Gene-to-pathway mapping.
    gene_to_pathway = {}
    for pw, genes in genesets.items():
        for g in genes:
            if g not in gene_to_pathway:
                gene_to_pathway[g] = []
            gene_to_pathway[g].append(pw)

    g2p_path = os.path.join(
        PROCESSED_DIR, "pathway_genesets", "gene_to_pathway.json"
    )
    with open(g2p_path, 'w', encoding='utf-8') as f:
        json.dump(gene_to_pathway, f, indent=2, ensure_ascii=False)
    print(f"  Gene-to-pathway: {g2p_path} ({len(gene_to_pathway)} genes)")

    # 3. Reactome hierarchy.
    build_reactome_hierarchy_json(parent_child_pairs)

    # 4. Align samples and save them.
    common_samples = pathway_scores.index.intersection(clinical_df.index)
    print(f"  Common samples: {len(common_samples)}")

    pathway_scores_aligned = pathway_scores.loc[common_samples]
    clinical_aligned = clinical_df.loc[common_samples]

    # Pathway-score matrix.
    X_path = os.path.join(
        PROCESSED_DIR, "expression_matrices", "tcga_pathway_scores.npy"
    )
    np.save(X_path, pathway_scores_aligned.values.astype(np.float32))
    print(f"  Pathway scores: {X_path} ({pathway_scores_aligned.shape})")

    # Survival data.
    times = clinical_aligned["time"].values.astype(np.float32)
    events = clinical_aligned["event"].values.astype(np.float32)

    np.save(os.path.join(PROCESSED_DIR, "tcga_times.npy"), times)
    np.save(os.path.join(PROCESSED_DIR, "tcga_events.npy"), events)
    print(f"  Survival: {len(times)} samples, {int(events.sum())} events")

    # Train/test split.
    is_train = clinical_aligned["is_train"].values.astype(bool)
    train_idx = np.where(is_train)[0]
    test_idx = np.where(~is_train)[0]

    np.save(os.path.join(PROCESSED_DIR, "tcga_train_idx.npy"), train_idx)
    np.save(os.path.join(PROCESSED_DIR, "tcga_test_idx.npy"), test_idx)
    print(f"  Train/Test split: {len(train_idx)}/{len(test_idx)}")

    # Save the sample-ID mapping.
    sample_ids = list(common_samples)
    sample_id_path = os.path.join(PROCESSED_DIR, "tcga_sample_ids.json")
    with open(sample_id_path, 'w', encoding='utf-8') as f:
        json.dump(sample_ids, f, indent=2, ensure_ascii=False)

    # 5. DAG matrices.
    dag_dir = os.path.join(PROCESSED_DIR, "baseline_dag")
    np.save(os.path.join(dag_dir, "baseline_adj.npy"), A_0)
    np.save(os.path.join(dag_dir, "enhanced_adj.npy"), A_enhanced)
    print(f"  Baseline DAG: {A_0.shape}, {int(A_0.sum())} edges")
    print(f"  Enhanced DAG: {A_enhanced.shape}, {int(A_enhanced.sum())} edges")


def create_rules_handbook(
    genesets: Dict[str, List[str]],
    pathway_names: List[str],
    mutations: Dict[str, List[Tuple[str, str]]],
) -> List[Dict]:
    """
    Create the rule handbook (Section 3.3 of the paper).

    Based on known HR+/HER2- breast cancer biology:
    - PI3K/AKT/mTOR-related pathways: PIK3CA mutation → activation.
    - ESR1-related pathways: ESR1 mutation → regulation.
    - Cell-cycle pathways: CCND1 amplification → activation.
    - Apoptosis pathways: TP53 mutation → inhibition.

    Identify key signaling axes from pathway names and establish low-level rules.
    """
    print(f"\n[7/7] Creating rules handbook...")

    rules = []

    # Key driver genes and regulatory directions.
    driver_rules = {
        "PIK3CA": {"direction": "activation", "delta": 1.5, "confidence": 0.90},
        "AKT1": {"direction": "activation", "delta": 1.4, "confidence": 0.85},
        "PTEN": {"direction": "inhibition", "delta": 0.5, "confidence": 0.85},
        "ESR1": {"direction": "activation", "delta": 1.3, "confidence": 0.80},
        "TP53": {"direction": "inhibition", "delta": 0.4, "confidence": 0.90},
        "MYC": {"direction": "activation", "delta": 1.6, "confidence": 0.75},
        "CCND1": {"direction": "activation", "delta": 1.4, "confidence": 0.80},
        "ERBB2": {"direction": "activation", "delta": 1.5, "confidence": 0.85},
        "MAPK1": {"direction": "activation", "delta": 1.3, "confidence": 0.75},
        "MAPK3": {"direction": "activation", "delta": 1.3, "confidence": 0.75},
        "GATA3": {"direction": "activation", "delta": 1.2, "confidence": 0.70},
        "FOXA1": {"direction": "activation", "delta": 1.2, "confidence": 0.70},
        "RB1": {"direction": "inhibition", "delta": 0.5, "confidence": 0.80},
        "NF1": {"direction": "inhibition", "delta": 0.6, "confidence": 0.75},
        "CDH1": {"direction": "inhibition", "delta": 0.7, "confidence": 0.70},
    }

    # Keyword-to-pathway mapping.
    keyword_map = {
        "pi3k": [], "akt": [], "mtor": [], "apoptosis": [],
        "cell_cycle": [], "dna_repair": [], "estrogen": [], "er": [],
        "signaling": [], "mapk": [], "wnt": [], "notch": [],
        "p53": [], "immune": [], "metabolism": [], "hypoxia": [],
    }

    for pw in pathway_names:
        pw_lower = pw.lower()
        for kw in keyword_map:
            if kw in pw_lower:
                keyword_map[kw].append(pw)

    # Match related pathways for each driver gene and generate rules.
    driver_gene_pathways = {}
    for gene, pw_list in genesets.items() if isinstance(genesets, dict) else []:
        for pw in pathway_names:
            if gene in genesets.get(pw, []):
                if gene not in driver_gene_pathways:
                    driver_gene_pathways[gene] = []
                driver_gene_pathways[gene].append(pw)

    # Generate rules directly by matching pathway_name keywords.
    for gene, config in driver_rules.items():
        # Find pathways containing this gene.
        related_pws = []
        for pw in pathway_names:
            if gene in genesets.get(pw, []):
                related_pws.append(pw)

        if not related_pws:
            continue

        # For each pathway sourced from this gene, find possible target pathways.
        source_pw = related_pws[0]  # Use the first matching pathway as the source.

        # Find target pathways (other signaling pathways).
        target_keywords = {
            "activation": ["apoptosis", "cell_cycle", "metabolism", "dna_repair"],
            "inhibition": ["apoptosis", "cell_cycle", "signaling"],
        }

        direction = config["direction"]
        targets = target_keywords.get(direction, ["apoptosis"])

        for kw in targets:
            target_pws = keyword_map.get(kw, [])
            for tgt_pw in target_pws[:2]:  # At most two rules per keyword.
                if tgt_pw != source_pw:
                    rules.append({
                        "source": source_pw,
                        "target": tgt_pw,
                        "gene": gene,
                        "delta": config["delta"] if direction == "activation" else -config["delta"],
                        "pmid": "REF",  # Placeholder; replace with an actual PMID.
                        "confidence": config["confidence"],
                    })

    # Deduplicate.
    seen = set()
    unique_rules = []
    for r in rules:
        key = (r["source"], r["target"], r["gene"])
        if key not in seen:
            seen.add(key)
            unique_rules.append(r)

    print(f"  Generated {len(unique_rules)} rules from {len(driver_rules)} driver genes")

    # Save.
    rules_path = os.path.join(BASE_DIR, "knowledge_engine", "rules_handbook.json")
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(unique_rules, f, indent=2, ensure_ascii=False)
    print(f"  Rules handbook saved to: {rules_path}")

    return unique_rules


def main():
    print("=" * 70)
    print("MuDAG-Pro Data Preprocessing (331 Pathways)")
    print("=" * 70)

    ensure_dirs()

    # 1. Load gene sets for 331 pathways.
    genesets = load_pathway_genesets()
    pathway_names = list(genesets.keys())

    # 2. Load RNA expression profiles.
    rna_path = os.path.join(RNA_DIR, "brca", "rna_clean.csv")
    print(f"\n  Loading RNA expression from: {rna_path}")
    expr_df = pd.read_csv(rna_path, index_col=0)
    print(f"  Expression: {expr_df.shape[0]} samples × {expr_df.shape[1]} genes")

    # 3. Compute pathway activation scores.
    pathway_scores = compute_pathway_scores(expr_df, genesets)

    # 4. Load clinical data.
    clinical_df = load_clinical_data()

    # 5. Build the DAG.
    A_0, parent_child_pairs = build_dag_from_reactome(pathway_names, genesets)

    # Enhanced DAG (baseline = enhanced initially; rules are integrated later).
    A_enhanced = A_0.copy()

    # 6. Load mutation data.
    mutations = load_mutation_data()

    # 7. Create the rule handbook.
    rules = create_rules_handbook(genesets, pathway_names, mutations)

    # 8. Save all preprocessed data.
    save_processed_data(
        genesets=genesets,
        pathway_scores=pathway_scores,
        clinical_df=clinical_df,
        A_0=A_0,
        A_enhanced=A_enhanced,
        parent_child_pairs=parent_child_pairs,
    )

    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"\n  Output directory: {PROCESSED_DIR}")
    print(f"  Pathways: {len(pathway_names)}")
    print(f"  TCGA samples: {len(pathway_scores)}")
    print(f"  DAG edges: {int(A_0.sum())}")
    print(f"  Rules: {len(rules)}")
    print(f"\nNext steps:")
    print(f"  python main.py --mode train")
    print(f"  python main.py --mode evaluate")


if __name__ == "__main__":
    main()
