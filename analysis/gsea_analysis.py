"""
MuDAG-Pro gene set enrichment analysis module (GSEA; Section 4.5 of the paper).

Performs differential gene-expression analysis between high- and low-risk groups,
then uses GSEA to identify significantly enriched pathways and biological processes.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class GSEAAnalyzer:
    """
    Gene set enrichment analyzer.

    Workflow:
    1. Differential gene-expression analysis between high- and low-risk groups.
    2. Gene ranking (by log2 fold change or t-statistic).
    3. Preranked GSEA.
    4. Visualization of significantly enriched gene sets.
    """

    def __init__(
        self,
        gene_set_libraries: Optional[List[str]] = None,
        n_permutations: int = 1000,
        fdr_threshold: float = 0.25,
        random_state: int = 42,
    ):
        """
        Args:
            gene_set_libraries: List of gene-set libraries.
                ["KEGG_2021_Human", "GO_Biological_Process_2021", "Reactome_2022"]
            n_permutations: Number of permutations.
            fdr_threshold: FDR significance threshold.
            random_state: Random seed.
        """
        self.gene_set_libraries = gene_set_libraries or [
            "KEGG_2021_Human",
            "GO_Biological_Process_2021",
            "Reactome_2022",
        ]
        self.n_permutations = n_permutations
        self.fdr_threshold = fdr_threshold
        self.random_state = random_state
        self.enrichment_results = {}

    def compute_differential_expression(
        self,
        expression_matrix: pd.DataFrame,
        risk_groups: np.ndarray,
        method: str = "ttest",
    ) -> pd.DataFrame:
        """
        Compute differential gene expression between high- and low-risk groups.

        Args:
            expression_matrix: Gene-expression matrix (genes x samples).
            risk_groups: Risk groups (0=low risk, 1=high risk).
            method: Statistical test method ("ttest", "welch", or "mannwhitney").

        Returns:
            deg_df: Differentially expressed gene DataFrame.
                columns: gene, log2fc, statistic, p_value, p_adj
        """
        high_mask = risk_groups == 1
        low_mask = risk_groups == 0

        genes = expression_matrix.index.tolist()
        n_genes = len(genes)

        log2fc_list = []
        stats_list = []
        pvals_list = []

        for gene in genes:
            high_vals = expression_matrix.loc[gene, high_mask].values.astype(float)
            low_vals = expression_matrix.loc[gene, low_mask].values.astype(float)

            # Log2 fold change
            mean_high = np.mean(high_vals)
            mean_low = np.mean(low_vals)

        # Prevent log2(0).
            if mean_high <= 0 and mean_low <= 0:
                log2fc = 0.0
            elif mean_high <= 0:
                log2fc = -10.0
            elif mean_low <= 0:
                log2fc = 10.0
            else:
                log2fc = np.log2(mean_high / mean_low)

        # Statistical tests.
            if method == "ttest":
                stat, pval = stats.ttest_ind(high_vals, low_vals, equal_var=True)
            elif method == "welch":
                stat, pval = stats.ttest_ind(high_vals, low_vals, equal_var=False)
            elif method == "mannwhitney":
                stat, pval = stats.mannwhitneyu(high_vals, low_vals, alternative='two-sided')
            else:
                stat, pval = 0.0, 1.0

            log2fc_list.append(log2fc)
            stats_list.append(stat)
            pvals_list.append(pval)

        # Multiple-testing correction (Benjamini-Hochberg).
        pvals = np.array(pvals_list)
        n = len(pvals)
        sorted_idx = np.argsort(pvals)
        p_adj = np.ones(n)
        for rank, idx in enumerate(sorted_idx):
            p_adj[idx] = min(pvals[idx] * n / (rank + 1), 1.0)
        # Preserve monotonicity.
        for i in range(n - 2, -1, -1):
            p_adj[sorted_idx[i]] = min(p_adj[sorted_idx[i]], p_adj[sorted_idx[i + 1]])

        deg_df = pd.DataFrame({
            "gene": genes,
            "log2fc": log2fc_list,
            "statistic": stats_list,
            "p_value": pvals_list,
            "p_adj": p_adj,
        })

        deg_df = deg_df.sort_values("p_value").reset_index(drop=True)
        return deg_df

    def run_gsea_preranked(
        self,
        deg_df: pd.DataFrame,
        gene_set_library: str = "KEGG_2021_Human",
    ) -> pd.DataFrame:
        """
        Perform preranked GSEA.

        Args:
            deg_df: Differentially expressed gene DataFrame (must contain gene and statistic columns).
            gene_set_library: Gene-set library name.

        Returns:
            gsea_results: GSEA enrichment-results DataFrame.
        """
        try:
            import gseapy as gp
        except ImportError:
            print("[GSEA] gseapy 未安装，使用内置简化实现")
            return self._run_gsea_simple(deg_df, gene_set_library)

            # Prepare the ranked gene list.
            # Use a signed statistic: sign(log2fc) * (-log10(p_value)).
        rnk = deg_df[["gene", "statistic"]].dropna().copy()
        rnk = rnk.sort_values("statistic", ascending=False)

        try:
            results = gp.prerank(
                rnk=rnk,
                gene_sets=gene_set_library,
                permutation_num=self.n_permutations,
                seed=self.random_state,
                threads=1,
                min_size=5,
                max_size=500,
                verbose=False,
            )

            self.enrichment_results[gene_set_library] = results
            return results.res2d
        except Exception as e:
            print(f"[GSEA] gseapy prerank 失败: {e}")
            return self._run_gsea_simple(deg_df, gene_set_library)

    def _run_gsea_simple(
        self, deg_df: pd.DataFrame, gene_set_library: str
    ) -> pd.DataFrame:
        """
        Simplified GSEA implementation (used when gseapy is unavailable).

        Uses Fisher's exact test for over-representation analysis (ORA).
        """
        # Define built-in gene sets (simplified).
        builtin_gene_sets = self._get_builtin_gene_sets(gene_set_library)

        if not builtin_gene_sets:
            print(f"[GSEA] 未找到内置基因集库 '{gene_set_library}'")
            return pd.DataFrame()

        # Define significantly differentially expressed genes (|log2FC| > 1, p_adj < 0.05).
        deg_sig = deg_df[
            (deg_df["p_adj"] < 0.05) & (abs(deg_df["log2fc"]) > 0.5)
        ]
        deg_genes = set(deg_sig["gene"].tolist())
        all_genes = set(deg_df["gene"].tolist())
        n_total = len(all_genes)

        results = []
        for gs_name, gs_genes in builtin_gene_sets.items():
            gs_set = set(gs_genes) & all_genes
            if len(gs_set) < 5:
                continue

            overlap = deg_genes & gs_set
            if len(overlap) == 0:
                continue

            # 2x2 contingency table.
            a = len(overlap)
            b = len(deg_genes) - a
            c = len(gs_set) - a
            d = n_total - a - b - c

            odds_ratio, p_value = stats.fisher_exact(
                [[a, b], [c, d]], alternative='greater'
            )

            # Enrichment score.
            enrichment_score = a / len(gs_set) if len(gs_set) > 0 else 0

            results.append({
                "Term": gs_name,
                "Gene_set_size": len(gs_set),
                "Overlap": a,
                "Odds_ratio": odds_ratio,
                "P_value": p_value,
                "Enrichment_score": enrichment_score,
                "Genes": "; ".join(sorted(overlap)[:10]),
            })

        result_df = pd.DataFrame(results)
        if len(result_df) > 0:
        # FDR correction.
            pvals = result_df["P_value"].values
            n = len(pvals)
            sorted_idx = np.argsort(pvals)
            fdr = np.ones(n)
            for rank, idx in enumerate(sorted_idx):
                fdr[idx] = min(pvals[idx] * n / (rank + 1), 1.0)
            result_df["FDR"] = fdr
            result_df = result_df[result_df["FDR"] < self.fdr_threshold]
            result_df = result_df.sort_values("FDR")

        return result_df

    def _get_builtin_gene_sets(self, library: str) -> Dict[str, List[str]]:
        """Return simplified built-in gene-set definitions for environments without gseapy."""
        # Core gene sets related to HR+/HER2- breast cancer.
        gene_sets = {
            "KEGG_2021_Human": {
                "PI3K-Akt signaling pathway": [
                    "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2", "AKT1", "AKT2",
                    "AKT3", "MTOR", "PTEN", "TSC1", "TSC2", "RICTOR",
                    "RPTOR", "EIF4EBP1", "RPS6KB1", "FOXO1", "FOXO3",
                ],
                "Estrogen signaling pathway": [
                    "ESR1", "ESR2", "PGR", "GPER1", "FOXA1", "GATA3",
                    "SP1", "CREBBP", "EP300", "NCOA1", "NCOA2", "NCOA3",
                    "NCOR1", "NCOR2", "FOS", "JUN",
                ],
                "Cell cycle": [
                    "CCND1", "CDK4", "CDK6", "RB1", "E2F1", "E2F2",
                    "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "TP53",
                    "MYC", "CCNE1", "CDK2",
                ],
                "MAPK signaling pathway": [
                    "KRAS", "HRAS", "NRAS", "BRAF", "RAF1", "MAP2K1",
                    "MAP2K2", "MAPK1", "MAPK3", "FOS", "JUN", "ELK1",
                ],
                "Apoptosis": [
                    "BCL2", "BAX", "BAK1", "BAD", "BID", "CASP3",
                    "CASP8", "CASP9", "CYCS", "APAF1", "MCL1", "BCL2L1",
                ],
                "mTOR signaling pathway": [
                    "MTOR", "RPTOR", "RICTOR", "AKT1S1", "DEPTOR",
                    "MLST8", "EIF4E", "EIF4EBP1", "RPS6KB1", "ULK1",
                ],
                "p53 signaling pathway": [
                    "TP53", "MDM2", "MDM4", "CDKN1A", "BAX", "BBC3",
                    "PMAIP1", "GADD45A", "RRM2B", "SESN1", "SESN2",
                ],
                "Wnt signaling pathway": [
                    "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B",
                    "TCF7L2", "LEF1", "CCND1", "MYC", "WNT1", "WNT3A",
                ],
                "Notch signaling pathway": [
                    "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1",
                    "JAG2", "DLL1", "DLL4", "RBPJ", "MAML1", "HES1",
                ],
                "TGF-beta signaling pathway": [
                    "TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
                    "SMAD4", "SMAD7", "BMPR1A", "BMPR2",
                ],
            },
            "GO_Biological_Process_2021": {
                "Positive regulation of cell proliferation": [
                    "CCND1", "MYC", "EGFR", "ERBB2", "AKT1", "MTOR",
                    "IGF1R", "FGF2", "PDGFRA", "KIT",
                ],
                "Negative regulation of apoptotic process": [
                    "BCL2", "MCL1", "BCL2L1", "XIAP", "BIRC5",
                    "AKT1", "HSPA1A", "HSPA1B",
                ],
                "Response to estrogen": [
                    "ESR1", "PGR", "GATA3", "TFF1", "GREB1",
                    "MYC", "CCND1", "FOXA1",
                ],
                "DNA damage response": [
                    "TP53", "ATM", "ATR", "BRCA1", "BRCA2",
                    "CHEK1", "CHEK2", "RAD51", "PARP1",
                ],
                "Epithelial to mesenchymal transition": [
                    "CDH1", "CDH2", "VIM", "SNAI1", "SNAI2",
                    "TWIST1", "ZEB1", "ZEB2", "TGFB1",
                ],
            },
            "Reactome_2022": {
                "Signaling by ERBB2": [
                    "ERBB2", "EGFR", "ERBB3", "PIK3CA", "AKT1",
                    "MTOR", "RPS6KB1",
                ],
                "PI3K/AKT Signaling in Cancer": [
                    "PIK3CA", "PIK3R1", "AKT1", "PTEN", "MTOR",
                    "TSC1", "TSC2",
                ],
                "Estrogen-dependent gene expression": [
                    "ESR1", "FOXA1", "GATA3", "PGR", "NCOA1",
                    "NCOA2", "EP300",
                ],
                "Cell Cycle Checkpoints": [
                    "TP53", "CDKN1A", "CDKN1B", "ATM", "ATR",
                    "CHEK1", "CHEK2", "CDC25A",
                ],
            },
        }

        return gene_sets.get(library, gene_sets.get("KEGG_2021_Human", {}))

    def plot_enrichment_heatmap(
        self,
        enrichment_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot an enriched-pathway heatmap.

        Args:
            enrichment_df: GSEA results DataFrame.
            top_n: Number of top significant entries to display.
            save_path: Save path.
            figsize: Figure size.

        Returns:
            fig: matplotlib Figure.
        """
        if enrichment_df.empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, 'No significant enrichment found',
                    ha='center', va='center', fontsize=14)
            return fig

        df = enrichment_df.head(top_n).copy()

        fig, ax = plt.subplots(figsize=figsize)

        # Color by -log10(FDR).
        neg_log_fdr = -np.log10(df["FDR"].values.clip(min=1e-10))
        enrichment = df["Enrichment_score"].values

        colors = plt.cm.RdYlBu_r(neg_log_fdr / max(neg_log_fdr.max(), 1))

        bars = ax.barh(
            range(len(df)),
            enrichment,
            color=colors,
            edgecolor='gray',
            linewidth=0.5,
        )

        # Color bar.
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdYlBu_r,
            norm=plt.Normalize(vmin=0, vmax=max(neg_log_fdr.max(), 1)),
        )
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('-log10(FDR)', fontsize=10)

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Term"].values, fontsize=9)
        ax.set_xlabel('Enrichment Score', fontsize=12)
        ax.set_title('GSEA Enrichment Results', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[GSEA] 富集热图已保存至: {save_path}")

        return fig

    def plot_enrichment_dotplot(
        self,
        enrichment_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot a GSEA bubble chart.

        Args:
            enrichment_df: GSEA results.
            top_n: Number of top entries.
            save_path: Save path.
            figsize: Figure size.

        Returns:
            fig: matplotlib Figure.
        """
        if enrichment_df.empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, 'No significant enrichment found',
                    ha='center', va='center', fontsize=14)
            return fig

        df = enrichment_df.head(top_n).copy()

        fig, ax = plt.subplots(figsize=figsize)

        neg_log_fdr = -np.log10(df["FDR"].values.clip(min=1e-10))
        gene_ratio = df["Overlap"].values / df["Gene_set_size"].values

        scatter = ax.scatter(
            gene_ratio,
            range(len(df)),
            s=df["Overlap"].values * 15 + 20,
            c=neg_log_fdr,
            cmap='RdYlBu_r',
            edgecolors='gray',
            linewidth=0.5,
            alpha=0.8,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('-log10(FDR)', fontsize=10)

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Term"].values, fontsize=9)
        ax.set_xlabel('Gene Ratio (Overlap / Gene Set Size)', fontsize=12)
        ax.set_title('GSEA Enrichment Dotplot', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # Size legend.
        for n_genes in [5, 10, 20]:
            ax.scatter(
                [], [],
                s=n_genes * 15 + 20,
                c='gray', edgecolors='gray',
                linewidth=0.5, alpha=0.6,
                label=f'{n_genes} genes',
            )
        ax.legend(
            title='Overlap size', loc='lower right',
            fontsize=8, title_fontsize=9,
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[GSEA] 气泡图已保存至: {save_path}")

        return fig
