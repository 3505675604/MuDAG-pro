"""
MuDAG-Pro survival-prediction dataset loader (Section 3.1 of the paper).

Handles conversion of gene-expression profiles into pathway activation scores,
loading of somatic mutation signatures, and wrapping data as PyTorch tensors.

Supports two input modes:
1. Gene-expression profile mode: raw gene-expression matrix → Z-score normalization → pathway scoring.
2. Precomputed pathway-score mode: directly load the computed pathway-score matrix X (N x M).

Equations from Section 3.1 of the paper:
    x_g(i) = (x_g(i) - μ_g) / σ_g                    (Z-score normalization)
    s_k(i) = (1/|G_k|) · Σ_{g∈G_k} x_g(i)           (pathway activation score)
    x_i = [s_1(i), ..., s_M(i)]^T ∈ R^{1×M}         (pathway feature vector)
"""
import os
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union

from src.pathway_scoring import (
    RobustPathwayScaler,
    compute_rank_pathway_scores,
)


class PathwayDataset(Dataset):
    """
    MuDAG-Pro survival-prediction dataset loader.

    Supports two input modes:
    - Gene-expression mode: expression_df is a (samples × genes) expression matrix.
    - Precomputed pathway mode: expression_df is a (samples × pathways) pathway-score matrix.
    """

    def __init__(
        self,
        expression_df: pd.DataFrame,
        pathway_genesets: Dict[str, List[str]],
        clinical_df: pd.DataFrame,
        mutation_dict: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        is_train: bool = True,
        train_gene_means: Optional[pd.Series] = None,
        train_gene_stds: Optional[pd.Series] = None,
        precomputed_pathways: bool = False,
        pathway_scoring_method: str = "zscore_mean",
        train_pathway_scaler: Optional[RobustPathwayScaler] = None,
    ):
        """
        Args:
            expression_df: Matrix with sample IDs as rows and gene/pathway symbols as columns.
            pathway_genesets: Dictionary mapping pathway names to their lists of genes.
            clinical_df: DataFrame containing survival endpoints; must include 'event' (0/1) and 'time' columns.
            mutation_dict: Optional dictionary mapping Sample_ID to a list of [(gene, variant)].
            is_train: Whether this is the training set (used to compute/apply Z-score means and standard deviations).
            train_gene_means: For a test set, gene means computed from the training set.
            train_gene_stds: For a test set, gene standard deviations computed from the training set.
            precomputed_pathways: If True, expression_df is already a pathway-score matrix and
                the Z-score normalization and pathway-scoring steps are skipped.
        """
        super().__init__()

        # 1. Ensure strict alignment of sample IDs.
        common_samples = expression_df.index.intersection(clinical_df.index)
        if len(common_samples) == 0:
            raise ValueError(
                "expression_df 与 clinical_df 没有重合的 Sample_ID，请检查索引设置！"
            )

        self.samples = list(common_samples)
        self.expression_df = expression_df.loc[self.samples].copy()
        self.clinical_df = clinical_df.loc[self.samples].copy()
        self.mutation_dict = mutation_dict if mutation_dict is not None else {}
        self.pathway_names = list(pathway_genesets.keys())
        self.precomputed_pathways = precomputed_pathways
        self.pathway_scoring_method = pathway_scoring_method
        self.pathway_scaler = train_pathway_scaler

        if precomputed_pathways:
            # External adapters emit raw rank-pathway scores. Apply only the
            # scaler fitted on TCGA training cases when one is supplied.
            self.pathway_scores_df = (
                train_pathway_scaler.transform(self.expression_df)
                if train_pathway_scaler is not None
                else self.expression_df
            )
            self.gene_means = None
            self.gene_stds = None
            self.normalized_expr = None
        elif pathway_scoring_method == "rank":
            raw_scores = compute_rank_pathway_scores(
                self.expression_df,
                pathway_genesets,
            )
            if is_train:
                self.pathway_scaler = RobustPathwayScaler.fit(raw_scores)
            elif train_pathway_scaler is None:
                raise ValueError(
                    "Rank-scored test data requires the TCGA training pathway scaler."
                )
            self.pathway_scores_df = self.pathway_scaler.transform(raw_scores)
            self.gene_means = None
            self.gene_stds = None
            self.normalized_expr = None
        elif pathway_scoring_method != "zscore_mean":
            raise ValueError(
                "pathway_scoring_method must be 'zscore_mean' or 'rank'."
            )
        else:
            # Gene-expression mode: Z-score normalization → pathway scoring.
            self.normalized_expr, self.gene_means, self.gene_stds = (
                self._normalize_expression(
                    is_train=is_train,
                    train_means=train_gene_means,
                    train_stds=train_gene_stds,
                )
            )
            self.pathway_scores_df = self._compute_pathway_scores(pathway_genesets)

        # Extract survival-endpoint tensors (time T and event Delta).
        self.times = torch.tensor(
            self.clinical_df['time'].values, dtype=torch.float32
        )
        self.events = torch.tensor(
            self.clinical_df['event'].values, dtype=torch.float32
        )

        # Convert to PyTorch tensor matrix X (N x M).
        self.X_tensor = torch.tensor(
            self.pathway_scores_df.values, dtype=torch.float32
        )

    def _normalize_expression(
        self,
        is_train: bool,
        train_means: Optional[pd.Series],
        train_stds: Optional[pd.Series],
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Apply Z-score normalization to gene expression (Section 3.1 of the paper):
        x_g = (x_g - μ_g) / σ_g

        Preserve the original zero for genes with Count=0 (not detected) to retain its biological meaning.
        """
        expr = self.expression_df.copy()
        zero_mask = (expr == 0)  # Record positions that were originally undetected zeros.

        if is_train:
            gene_means = expr.mean(axis=0)
            gene_stds = expr.std(axis=0)
            # Prevent division by zero (replace zero standard deviations with 1).
            gene_stds = gene_stds.replace(0, 1.0)
        else:
            if train_means is None or train_stds is None:
                raise ValueError(
                    "测试集必须传入训练集的 train_gene_means 和 train_gene_stds！"
                )
            gene_means = train_means
            gene_stds = train_stds

        # Apply Z-score standardization.
        norm_expr = (expr - gene_means) / gene_stds

        # Strictly follow Section 3.1 of the paper: retain the original zero for undetected values (count=0).
        norm_expr[zero_mask] = 0.0

        return norm_expr, gene_means, gene_stds

    def _compute_pathway_scores(
        self, pathway_genesets: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Compute the sample's baseline activation scores across M core pathways (Section 3.1 of the paper):
        s_k = (1 / |G_k|) · Σ_{g∈G_k} x_g

        Use the arithmetic mean (low-pass filtering) to smooth gene-level noise.
        """
        scores = {}
        for p_name in self.pathway_names:
            genes_in_pathway = pathway_genesets[p_name]
            # Find the subset of pathway genes actually present in the current cohort data.
            available_genes = [
                g for g in genes_in_pathway
                if g in self.normalized_expr.columns
            ]

            if len(available_genes) > 0:
                pathway_score = self.normalized_expr[available_genes].mean(axis=1)
            else:
                pathway_score = pd.Series(0.0, index=self.samples)

            scores[p_name] = pathway_score

        return pd.DataFrame(scores, index=self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, str, List[Tuple[str, str]]]]:
        sample_id = self.samples[idx]

        return {
            "sample_id": sample_id,
            "x_base": self.X_tensor[idx],           # Pathway activation-score vector x_i.
            "time": self.times[idx],                 # Survival time T_i.
            "event": self.events[idx],               # Event status δ_i.
            "mutation_profile": self.mutation_dict.get(sample_id, []),  # S_i
        }

    def get_pathway_matrix(self) -> pd.DataFrame:
        """Return the complete pathway-score DataFrame."""
        return self.pathway_scores_df


# ==============================================================================
# Helper data-loading functions.
# ==============================================================================

def load_pathway_genesets(json_path: str) -> Dict[str, List[str]]:
    """Load the pathway-to-gene-set mapping from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        genesets = json.load(f)
    return genesets


def load_mutation_signatures(
    mutation_json_path: str,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load patient somatic mutation signatures S_i filtered by OncoKB.

    Supports two formats:
    1. {"Sample_1": [["PIK3CA", "H1047R"], ["ESR1", "Y537S"]], ...}
    2. {"Sample_1": [["PIK3CA", "missense_variant"], ...], ...}
    """
    if not os.path.exists(mutation_json_path):
        print(
            f"Warning: 突变文件 {mutation_json_path} 不存在，"
            f"将返回空字典 (退化为静态模型)。"
        )
        return {}
    with open(mutation_json_path, 'r', encoding='utf-8') as f:
        raw_muts = json.load(f)

    # Convert to the standard [(gene, variant)] tuple format. TCGA mutation
    # files are aliquot-level while expression/clinical data are patient-level,
    # so collapse only valid TCGA barcodes to their first three components.
    tcga_patient_pattern = re.compile(
        r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-|$)", re.IGNORECASE
    )
    mutation_signatures = {}
    seen_mutations = {}
    for sample, mut_list in raw_muts.items():
        if not isinstance(mut_list, list) or not mut_list:
            continue

        sample_id = str(sample).strip()
        if tcga_patient_pattern.match(sample_id):
            sample_id = "-".join(sample_id.upper().split("-")[:3])

        mutation_signatures.setdefault(sample_id, [])
        seen_mutations.setdefault(sample_id, set())

        if isinstance(mut_list[0], list):
            standardized = [
                (
                    str(item[0]).strip().upper(),
                    str(item[1]).strip() if len(item) > 1 else "",
                )
                for item in mut_list
                if item
            ]
        elif isinstance(mut_list[0], str):
            standardized = [(str(g).strip().upper(), "") for g in mut_list]
        else:
            standardized = []

        for mutation in standardized:
            if mutation not in seen_mutations[sample_id]:
                mutation_signatures[sample_id].append(mutation)
                seen_mutations[sample_id].add(mutation)
    return mutation_signatures


def load_pathway_scores_from_npy(
    npy_path: str,
    sample_ids: Optional[List[str]] = None,
    pathway_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a precomputed pathway activation-score matrix from an npy file.

    Args:
        npy_path: Path to the npy file.
        sample_ids: List of sample IDs (index).
        pathway_names: List of pathway names (column names).

    Returns:
        pathway_df: (samples × pathways) DataFrame.
    """
    X = np.load(npy_path)
    if pathway_names is None:
        pathway_names = [f"Pathway_{i}" for i in range(X.shape[1])]
    if sample_ids is None:
        sample_ids = [f"Sample_{i}" for i in range(X.shape[0])]

    return pd.DataFrame(X, index=sample_ids, columns=pathway_names)
