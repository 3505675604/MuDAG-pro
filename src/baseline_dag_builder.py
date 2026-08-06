import os
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional


class BaselineDAGBuilder:
    """
    MuDAG-Pro population-level and enhanced DAG builder.
    Implements transitive-closure computation, LLM rule integration, and topological cycle filtering with T_acyclic.
    """
    def __init__(self, pathway_names: List[str]):
        """
        Args:
            pathway_names: Names of the M=331 core pathways, used to fix the matrix row and column order.
        """
        self.pathway_names = pathway_names
        self.M = len(pathway_names)
        self.p2idx = {name: i for i, name in enumerate(pathway_names)}

    def build_baseline_dag(
        self,
        parent_child_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the baseline DAG adjacency matrix from Reactome parent-child relationships (Section 3.2 of the paper).

        Args:
            parent_child_pairs: Direct hierarchical edges in the form [(parent_pathway, child_pathway), ...].

        Returns:
            B: Boolean direct-adjacency matrix B in {0, 1}^(M x M).
            A_0: Transitive-closure baseline adjacency matrix without self-loops, A_0 in {0, 1}^(M x M).
        """
        # 1. Initialize the direct boolean adjacency matrix B.
        B = np.zeros((self.M, self.M), dtype=np.float32)
        for parent, child in parent_child_pairs:
            if parent in self.p2idx and child in self.p2idx:
                u, v = self.p2idx[parent], self.p2idx[child]
                if u != v:  # Exclude self-loops.
                    B[u, v] = 1.0

        # 2. Compute the transitive closure to derive reachability matrix W (Equation 3.2 in the paper).
        # W = \bigcup_{d=1}^{M-1} B^d
        # Use an efficient Warshall implementation under boolean/0-1 matrix multiplication.
        W = (B > 0).astype(bool)
        for k in range(self.M):
            # If i can reach k and k can reach j, then i can reach j.
            W = W | (W[:, [k]] & W[[k], :])

        W_float = W.astype(np.float32)

        # 3. Remove self-loops to form the population-level baseline pathway DAG: A_0 = W - diag(W).
        np.fill_diagonal(W_float, 0.0)
        A_0 = W_float

        return B, A_0

    def filter_acyclic(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Topological cycle detection and filtering operator T_acyclic(·) - diag(·) (Section 3.4 of the paper).
        Uses Kahn's algorithm for topological sorting; when a cycle is found, removes an edge by priority/minimum weight to break it.

        Args:
            adj_matrix: M x M adjacency matrix that may contain cycles or self-loops.

        Returns:
            A_dag: Adjacency matrix that strictly satisfies the self-loop-free directed acyclic graph (DAG) constraints.
        """
        A = adj_matrix.copy()

        # 1. Force removal of the diagonal (diag(·)).
        np.fill_diagonal(A, 0.0)

        # 2. Iteratively detect and remove cycles to enforce the directed acyclic graph (DAG) constraint.
        while True:
            # Compute the in-degree of every node in the current graph.
            # A[u, v] > 0 indicates an edge u -> v, so column sums give the in-degrees.
            in_degrees = (A > 0).sum(axis=0)

            # Use Kahn's algorithm to find a topological ordering.
            queue = [i for i in range(self.M) if in_degrees[i] == 0]
            visited_count = 0
            temp_in_degrees = in_degrees.copy()

            while queue:
                u = queue.pop(0)
                visited_count += 1
                for v in range(self.M):
                    if A[u, v] > 0:
                        temp_in_degrees[v] -= 1
                        if temp_in_degrees[v] == 0:
                            queue.append(v)

            # If every node is visited, the graph is acyclic; exit the loop.
            if visited_count == self.M:
                break

            # Otherwise, a cycle exists. Find the set of nodes involved in cycles.
            # Nodes not visited by the topological sort are part of cycles.
            unvisited_nodes = np.where(temp_in_degrees > 0)[0]

            # Remove the minimum-weight edge on a cycle (cycle-breaking strategy).
            min_weight = float('inf')
            edge_to_remove = None

            for u in unvisited_nodes:
                for v in unvisited_nodes:
                    if A[u, v] > 0 and A[u, v] < min_weight:
                        min_weight = A[u, v]
                        edge_to_remove = (u, v)

            if edge_to_remove is not None:
                u, v = edge_to_remove
                u_name, v_name = self.pathway_names[u], self.pathway_names[v]
                print(f"[DAG Check] 检测到拓扑环路，自动删除边: {u_name} -> {v_name} (Weight: {min_weight:.4f})")
                A[u, v] = 0.0
            else:
                break

        return A

    def build_enhanced_dag(
        self,
        A_0: np.ndarray,
        llm_rules: List[Dict]
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        """
        Integrate LLM-mined rules R_1 into the baseline DAG to build enhanced population-level DAG A (Section 3.4 of the paper).

        Equations from the paper:
            W_jk^new = Σ_{r∈R(j,k)} δ_r · c_r
            A = P(A_0 + W_new) = T_acyclic(A_0 + W_new) - diag(A_0 + W_new)

        Here δ_r may represent either direction of regulation:
        - δ_r > 1: Activating regulation (increases edge weight).
        - 0 < δ_r < 1: Inhibitory regulation (decreases edge weight).

        Args:
            A_0: Baseline adjacency matrix A_0 in R^(M x M).
            llm_rules: Global rule set R_1, formatted as:
                [{"source": "Pathway_J", "target": "Pathway_K",
                  "delta": 1.5, "confidence": 0.8, "gene": "PIK3CA"}, ...]

        Returns:
            A: Enhanced population-level DAG adjacency matrix A in R^(M x M).
            W_new_dict: Weight dictionary for added/modified edges, {(u, v): (weight, n_rules)}.
        """
        W_new = np.zeros((self.M, self.M), dtype=np.float32)
        edge_rule_accum = {}
        edge_rule_count = {}

        # 1. Iterate over all rules in R_1 and accumulate W_jk^new = Σ δ_r · c_r (Section 3.4 of the paper).
        for rule in llm_rules:
            src = rule.get("source", "")
            tgt = rule.get("target", "")
            delta = float(rule.get("delta", 0.0))
            c = float(rule.get("confidence", 0.5))

            if src in self.p2idx and tgt in self.p2idx:
                u, v = self.p2idx[src], self.p2idx[tgt]
                if u == v:
                    continue

                if (u, v) not in edge_rule_accum:
                    edge_rule_accum[(u, v)] = 0.0
                    edge_rule_count[(u, v)] = 0

                edge_rule_accum[(u, v)] += delta * c
                edge_rule_count[(u, v)] += 1

        # Apply the accumulated weights to W_new.
        for (u, v), val in edge_rule_accum.items():
            W_new[u, v] = val
            u_name = self.pathway_names[u]
            v_name = self.pathway_names[v]
            n_rules = edge_rule_count[(u, v)]
            direction = "↑" if val > 0 else "↓"
            print(f"[DAG] 规则边: {u_name} → {v_name} "
                  f"W_new={val:+.3f} ({n_rules} rules) {direction}")

        # 2. Integrate the baseline matrix: A_raw = A_0 + W_new (Section 3.4 of the paper).
        A_raw = A_0 + W_new

        # 3. Apply the topological acyclicity operator: A = P(A_raw) = T_acyclic(A_raw) - diag(A_raw).
        A = self.filter_acyclic(A_raw)

        # Build the rule dictionary.
        W_new_dict = {
            (u, v): (float(edge_rule_accum[(u, v)]), edge_rule_count[(u, v)])
            for (u, v) in edge_rule_accum
        }

        return A, W_new_dict


# ==============================================================================
# Helper loading and export functions.
# ==============================================================================

def load_reactome_hierarchy(json_path: str) -> List[Tuple[str, str]]:
    """Load the Reactome parent-child hierarchical edge list from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expected format: [["Parent_Pathway", "Child_Pathway"], ...].
    return [tuple(pair) for pair in data]


def save_adj_matrix(adj_matrix: np.ndarray, filepath: str):
    """Save an adjacency matrix as an npy array."""
    np.save(filepath, adj_matrix)


def load_adj_matrix(filepath: str) -> np.ndarray:
    """Load an adjacency matrix from an npy array."""
    return np.load(filepath)
