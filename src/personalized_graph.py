"""
MuDAG-Pro personalized adjustment-matrix builder for patients awaiting prediction (Section 3.5 of the paper).

During online inference, the LLM is completely removed from the computation chain; only the frozen rule handbook R_1 is used.
Matching rules are retrieved from the mutation signature S_i of the patient awaiting prediction,
a patient-specific sparse edge-adjustment matrix Γ_i is synthesized,
and the patient-specific adjacency matrix is built with the Hadamard product A_i = A ⊙ Γ_i.

Mathematical formula:
    Γ_i,jk = clip(1 + Σ_{r∈R_i(j,k)} δ_r · c_r, τ_min, τ_max)  if (j,k) ∈ E
             0                                                    otherwise

Where:
    R_i(j,k) = Retrieve(S_i, (j,k), R_1) — retrieve matching rules from the patient's mutation features.
    The constant 1 represents the baseline DAG edge weight without mutation perturbation.
    R_i(j,k) = ∅ means the patient carries no known mutation affecting this edge, so its weight remains unchanged.
    clip(x, a, b) = max(a, min(x, b)) is a two-sided clipping operator.
"""
import json
import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set


class PersonalizedGraphModulator:
    """
    MuDAG-Pro personalized adjustment-matrix builder for patients awaiting prediction (Section 3.5 of the paper).
    Matches the offline rule handbook R_1 against mutation signature S_i for a patient awaiting prediction,
    then synthesizes the patient-specific adjacency matrix through the Hadamard product A_i = A ⊙ Γ_i.
    """
    def __init__(
        self,
        pathway_names: List[str],
        rules_handbook: List[Dict],
        gene2pathway_map: Dict[str, List[str]],
        tau_min: float = 0.1,
        tau_max: float = 5.0
    ):
        """
        Args:
            pathway_names: List of M=331 core pathway names.
            rules_handbook: Rule handbook R_1 refined offline by an LLM and then frozen.
                Each rule has the format: {
                    "source": "Pathway_J",
                    "target": "Pathway_K",
                    "gene": "PIK3CA",           # Trigger gene corresponding to the rule.
                    "delta": 1.5,               # Adjustment multiplier.
                    "confidence": 0.85,         # Literature confidence.
                    "pmid": "12345678",
                    "regulation_type": "activates"
                }
            gene2pathway_map: Dictionary mapping OncoKB genes to Reactome pathway nodes.
                {"PIK3CA": ["REACTOME_PI3K_AKT_SIGNALING", ...], ...}
            tau_min: Lower bound for signal inhibition (default 0.1).
            tau_max: Upper bound for signal activation amplification (default 5.0).
        """
        self.pathway_names = pathway_names
        self.M = len(pathway_names)
        self.p2idx = {name: i for i, name in enumerate(pathway_names)}
        self.rules_handbook = rules_handbook
        self.gene2pathway = gene2pathway_map
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Build a fast rule-lookup index: {(src_idx, tgt_idx): [rule_1, rule_2, ...]}.
        self.rule_index = self._build_rule_index()

        # Build a reverse gene-to-rule-edge index.
        self.gene_edge_index: Dict[str, List[Tuple[int, int]]] = {}
        self._build_gene_edge_index()

    def _build_rule_index(self) -> Dict[Tuple[int, int], List[Dict]]:
        """Build a fast mapping from (source pathway, target pathway) to a rule subset."""
        index: Dict[Tuple[int, int], List[Dict]] = {}
        for rule in self.rules_handbook:
            src = rule.get("source", "")
            tgt = rule.get("target", "")
            if src in self.p2idx and tgt in self.p2idx:
                u, v = self.p2idx[src], self.p2idx[tgt]
                if u == v:
                    continue
                if (u, v) not in index:
                    index[(u, v)] = []
                index[(u, v)].append(rule)
        return index

    def _build_gene_edge_index(self):
        """Build a fast index from genes to related rule edges."""
        for (u, v), rules in self.rule_index.items():
            for rule in rules:
                gene = rule.get("gene", None)
                if gene is not None:
                    if gene not in self.gene_edge_index:
                        self.gene_edge_index[gene] = []
                    self.gene_edge_index[gene].append((u, v))

    def build_patient_modulation_matrix(
        self,
        mutation_profile: List[Tuple[str, str]],
        base_adj_mask: np.ndarray
    ) -> np.ndarray:
        """
        Synthesize sparse edge-adjustment matrix Γ_i from mutation signature S_i for patient i awaiting prediction (Equation 3.5 in the paper).

        Algorithm (mutation-pathway-edge mapping in Section 3.3 of the paper):
        1. Extract the set of genes G_i carrying functional mutations from S_i.
        2. Map G_i to the set of Reactome pathway nodes P_i^mut that they regulate.
        3. From the enhanced DAG edge list, select all outgoing edges whose source is in P_i^mut.
        4. Retrieve matching rules and calculate adjustment coefficients.

        Args:
            mutation_profile: Patient mutation signature S_i = [(gene, variant), ...].
            base_adj_mask: Binary topology mask for the enhanced DAG (True where A > 0).

        Returns:
            Gamma_i: Personalized edge-adjustment matrix Γ_i with shape M x M.
        """
        # Default to an all-ones matrix (edge weights remain 1.0 without mutation perturbation).
        Gamma_i = np.ones((self.M, self.M), dtype=np.float32)

        if not mutation_profile:
            # With missing mutation data, use an all-ones adjustment matrix so the model falls back to a population-level static model (a key design property from the paper).
            return Gamma_i

        # 1. Extract the set of genes G_i carrying functional mutations from S_i (Section 3.3 of the paper).
        mutated_genes = set([item[0] for item in mutation_profile])

        # 2. Map G_i to the Reactome pathway nodes P_i^mut they regulate (Step ii in Section 3.3 of the paper).
        P_mut: Set[str] = set()
        for gene in mutated_genes:
            if gene in self.gene2pathway:
                P_mut.update(self.gene2pathway[gene])

        # Convert to a set of pathway indices.
        P_mut_idx: Set[int] = set([
            self.p2idx[p] for p in P_mut if p in self.p2idx
        ])

        # 3. Iterate over candidate edges, match rules, and calculate adjustment coefficients (Section 3.5 of the paper).
        for (u, v), rules in self.rule_index.items():
            # Paper constraint: adjust only when u belongs to P_i^mut and the edge exists in the enhanced DAG.
            if u not in P_mut_idx:
                continue
            if not base_adj_mask[u, v]:
                continue

            delta_sum = 0.0
            matched_rules_count = 0

            for r in rules:
                # Verify that the rule's gene is actually present in the patient's mutation signature (Retrieve operator in Section 3.5 of the paper).
                rule_gene = r.get("gene", None)
                if rule_gene is None or rule_gene in mutated_genes:
                    # Equation from the paper: Σ δ_r · c_r.
                    delta_r = r.get("delta", 1.0)
                    c_r = r.get("confidence", 1.0)
                    delta_sum += delta_r * c_r
                    matched_rules_count += 1

            if matched_rules_count > 0:
                # Equation from the paper: Γ_i,jk = clip(1 + Σ δ_r · c_r, τ_min, τ_max).
                raw_val = 1.0 + delta_sum
                # Apply the two-sided clipping operator clip(x, τ_min, τ_max).
                Gamma_i[u, v] = np.clip(raw_val, self.tau_min, self.tau_max)

        return Gamma_i

    def get_patient_adjacency(
        self,
        A_enhanced: np.ndarray,
        mutation_profile: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Synthesize patient-specific pathway adjacency matrix A_i = A ⊙ Γ_i (Section 3.6 of the paper).

        Args:
            A_enhanced: Enhanced population-level DAG adjacency matrix A (M x M).
            mutation_profile: Patient mutation signature S_i.

        Returns:
            A_i: Patient-specific adjacency matrix A_i (M x M).
        """
        # Identify edges present in the enhanced DAG.
        base_mask = (A_enhanced > 0)
        # Build the personalized adjustment matrix.
        Gamma_i = self.build_patient_modulation_matrix(mutation_profile, base_mask)
        # Hadamard product (element-wise multiplication).
        A_i = A_enhanced * Gamma_i
        return A_i

    def get_matched_rules_for_patient(
        self,
        mutation_profile: List[Tuple[str, str]]
    ) -> List[Dict]:
        """
        Return a summary of rules matched by the patient's mutation signature (for white-box interpretation and report generation).

        Args:
            mutation_profile: Patient mutation signature S_i.

        Returns:
            matched: List of matched rules, including pathway names and PMID provenance.
        """
        if not mutation_profile:
            return []

        mutated_genes = set([item[0] for item in mutation_profile])
        matched = []

        for (u, v), rules in self.rule_index.items():
            for rule in rules:
                rule_gene = rule.get("gene", None)
                if rule_gene is None or rule_gene in mutated_genes:
                    matched.append({
                        "source_pathway": self.pathway_names[u],
                        "target_pathway": self.pathway_names[v],
                        "regulation_type": rule.get("regulation_type", "modulates"),
                        "delta": rule.get("delta", 1.0),
                        "confidence": rule.get("confidence", 0.5),
                        "pmid": rule.get("pmid", ""),
                        "matched_gene": rule_gene,
                    })

        return matched


# ==============================================================================
# Helper functions.
# ==============================================================================

def load_rules_handbook(json_path: str) -> List[Dict]:
    """Load the globally frozen rule handbook R_1 from a JSON file."""
    if not os.path.exists(json_path):
        print(f"Warning: 规则手册 {json_path} 不存在，返回空列表 (退化为静态模型)")
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    return rules


def load_gene2pathway_map(json_path: str) -> Dict[str, List[str]]:
    """Load the mapping from OncoKB genes to Reactome pathways."""
    if not os.path.exists(json_path):
        print(f"Warning: 基因-通路映射 {json_path} 不存在")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        g2p = json.load(f)
    return g2p


import os  # noqa: E402 (moved to top for readability but kept here for load functions)
