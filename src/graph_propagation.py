"""
MuDAG-Pro single-step directed pathway signal-propagation operator (Section 3.6 of the paper).

Explicitly models downstream cascading of upstream pathway signals along mutation-enhanced directed edges.

Single-sample propagation:
    x̃_i = x_i + α · x_i · A_i

Batch propagation (population-level static):
    X̃ = X + α · X · A

Batch propagation (personalized dynamic):
    X̃ = X + α · X · A_batch  (each sample uses a different A_i)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union, Tuple


class DirectedPathwayPropagation(nn.Module):
    """
    MuDAG-Pro single-step directed pathway signal-propagation operator (Section 3.6 of the paper).
    Explicitly models downstream cascading of upstream pathway signals along mutation-enhanced directed edges.
    """
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Global propagation-strength hyperparameter α (balances aggregated neighborhood signals and baseline scores).
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)

    def forward_single(
        self,
        x_base: torch.Tensor,
        A_i: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate pathway signals for a single sample (Equation 3.6 in the paper):
        \tilde{x}_i = x_i + \alpha \cdot x_i A_i

        Args:
            x_base: Single-sample baseline pathway-score vector x_i (shape: 1 x M or M).
            A_i: Patient-specific pathway adjacency matrix A_i (shape: M x M).

        Returns:
            x_tilde: Propagation-enhanced pathway representation \tilde{x}_i (same shape as x_base).
        """
        if x_base.dim() == 1:
            x_base = x_base.unsqueeze(0)  # (1, M)

        # Message-passing operator: x_i A_i performs weighted downstream aggregation of upstream node scores along directed edges.
        message = torch.matmul(x_base, A_i)  # (1, M)
        x_tilde = x_base + self.alpha * message
        return x_tilde

    def forward_batch_static(
        self,
        X_base: torch.Tensor,
        A_static: torch.Tensor
    ) -> torch.Tensor:
        """
        Population-level static DAG pathway signal propagation (for batched acceleration when mutation data is missing, or M_static ablation):
        \tilde{X} = X + \alpha \cdot X A

        Args:
            X_base: Cohort baseline pathway-feature matrix X (shape: N x M).
            A_static: Population-level static adjacency matrix A (shape: M x M).

        Returns:
            X_tilde: Batch propagation-enhanced feature matrix \tilde{X} (shape: N x M).
        """
        message = torch.matmul(X_base, A_static)  # (N, M)
        X_tilde = X_base + self.alpha * message
        return X_tilde

    def forward_batch_personalized(
        self,
        X_base: torch.Tensor,
        A_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Personalized dynamic DAG pathway signal propagation (supports batched computation):
        \tilde{x}_i = x_i + \alpha \cdot x_i A_i

        Args:
            X_base: Cohort/batch baseline pathway-feature matrix X (shape: B x M).
            A_batch: Tensor of patient-specific adjacency matrices within the batch, A_{batch} (shape: B x M x M).

        Returns:
            X_tilde: Propagation-enhanced batch feature matrix (shape: B x M).
        """
        B, M = X_base.shape
        # X_base: (B, 1, M)
        X_unsqueeze = X_base.unsqueeze(1)
        # bmm batch matrix multiplication: (B, 1, M) x (B, M, M) -> (B, 1, M).
        message = torch.bmm(X_unsqueeze, A_batch).squeeze(1)  # (B, M)
        X_tilde = X_base + self.alpha * message
        return X_tilde


def batch_build_patient_adjacencies(
    modulator: "PersonalizedGraphModulator",
    A_enhanced: np.ndarray,
    mutation_profiles: List[List[Tuple[str, str]]]
) -> torch.Tensor:
    """
    Helper function: generate A_i for every sample in a batch and stack them into a PyTorch tensor (B x M x M).

    Args:
        modulator: PersonalizedGraphModulator instance.
        A_enhanced: Enhanced population-level DAG adjacency matrix A (M x M).
        mutation_profiles: List of mutation signatures for patients in the batch.

    Returns:
        A_batch_tensor: Batch adjacency-matrix tensor (B x M x M).
    """
    adj_list = []
    for mut_prof in mutation_profiles:
        A_i = modulator.get_patient_adjacency(A_enhanced, mut_prof)
        adj_list.append(A_i)

    A_batch_np = np.stack(adj_list, axis=0)  # (B, M, M)
    return torch.tensor(A_batch_np, dtype=torch.float32)


def normalize_adjacency_by_outdegree(adj: np.ndarray) -> np.ndarray:
    """
    Normalize the adjacency matrix by out-degree so each row sums to 1 (nodes with zero in-degree remain all-zero).

    Args:
        adj: Original adjacency matrix A in R^(M x M).

    Returns:
        adj_norm: Out-degree-normalized adjacency matrix.
    """
    out_degrees = adj.sum(axis=1, keepdims=True)
    # Avoid division by zero.
    out_degrees = np.where(out_degrees > 0, out_degrees, 1.0)
    adj_norm = adj / out_degrees
    return adj_norm


def build_edge_list(adj_matrix: np.ndarray, pathway_names: List[str]) -> List[Tuple[str, str, float]]:
    """
    Build a sparse triplet edge list from an adjacency matrix (Section 3.4 of the paper).

    Args:
        adj_matrix: Adjacency matrix A in R^(M x M).
        pathway_names: List of pathway names.

    Returns:
        edge_list: [(source_pathway, target_pathway, weight), ...].
    """
    edges = []
    M = adj_matrix.shape[0]
    for u in range(M):
        for v in range(M):
            if u != v and adj_matrix[u, v] > 0:
                edges.append((pathway_names[u], pathway_names[v], float(adj_matrix[u, v])))
    return edges
