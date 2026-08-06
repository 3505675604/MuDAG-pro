"""
MuDAG-Pro Kaplan-Meier survival analysis module (reproduces Figure 4 and Section 4.6 of the paper).

Features:
- Split high- and low-risk groups at the median predicted PI.
- Plot Kaplan-Meier survival curves.
- Perform a two-sided log-rank test.
- Compute HR and 95% CI with a univariate Cox proportional hazards model.

Equations from Section 4.6 of the paper:
    X_i = 1 if PI_i >= Median(PI)  (high-risk group)
    X_i = 0 if PI_i <  Median(PI)  (low-risk group)

    h_i(t) = h_0(t) · exp(γ · X_i)
    HR = exp(γ)

    ℓ(γ) = Σ_k [γ·X_{(k)} - ln(Σ_{j∈R_k} exp(γ·X_j))]
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def plot_kaplan_meier(
    predictions_csv: str,
    dataset_name: str,
    endpoint_label: str = "OS",
    output_dir: str = "outputs/figures",
    save_format: str = "png",
) -> Dict:
    """
    Plot Kaplan-Meier survival curves, perform a log-rank test, and calculate the univariate HR
    (Section 4.6 and Figure 4 of the paper).

    Args:
        predictions_csv: Path to the predictions CSV produced by evaluation.
            Must contain the columns sample_id, predicted_pi, time, and event.
        dataset_name: Dataset name (for example, "TCGA-BRCA", "METABRIC", "SCAN-B", or "GEO").
        endpoint_label: Clinical endpoint label ("OS" = overall survival, "DMFS" = distant metastasis-free survival).
        output_dir: Chart output directory.
        save_format: Save format ("png" or "pdf").

    Returns:
        results: Dictionary containing p_value, hr, ci_lower, and ci_upper.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(predictions_csv)

    # ================================================================
    # 1. Split into high- and low-risk groups at the median prognostic index (PI) (Equation 4.6 in the paper).
    # ================================================================
    median_pi = df["predicted_pi"].median()
    df["risk_group"] = (df["predicted_pi"] >= median_pi).astype(int)

    high_risk_df = df[df["risk_group"] == 1]
    low_risk_df = df[df["risk_group"] == 0]

    n_high = len(high_risk_df)
    n_low = len(low_risk_df)

    # ================================================================
    # 2. Perform a two-sided log-rank test (Section 4.6 of the paper).
    # ================================================================
    lr_results = logrank_test(
        high_risk_df["time"], low_risk_df["time"],
        event_observed_A=high_risk_df["event"],
        event_observed_B=low_risk_df["event"],
    )
    p_value = lr_results.p_value

    # ================================================================
    # 3. Fit a univariate Cox proportional hazards model to compute HR and 95% CI (Section 4.6 of the paper).
    # ================================================================
    cph = CoxPHFitter()
    cph.fit(
        df[["time", "event", "risk_group"]],
        duration_col="time",
        event_col="event",
        show_progress=False,
    )
    hr = cph.hazard_ratios_["risk_group"]
    ci_lower, ci_upper = cph.confidence_intervals_.loc["risk_group"].values

    # ================================================================
    # 4. Plot Kaplan-Meier survival curves (Figure 4 of the paper).
    # ================================================================
    plt.figure(figsize=(7, 6))
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    kmf_low.fit(
        low_risk_df["time"], low_risk_df["event"],
        label=f"Low risk (n={n_low})",
    )
    kmf_high.fit(
        high_risk_df["time"], high_risk_df["event"],
        label=f"High risk (n={n_high})",
    )

    ax = plt.subplot(111)
    # Paper color scheme: blue for the low-risk group and orange-red for the high-risk group.
    kmf_low.plot_survival_function(ax=ax, color="#2b5c8f", ci_alpha=0.15)
    kmf_high.plot_survival_function(ax=ax, color="#d95f02", ci_alpha=0.15)

    # Annotate the log-rank p-value and HR information.
    p_str = "Log-rank p < 0.0001" if p_value < 0.0001 else f"Log-rank p = {p_value:.4f}"
    hr_str = f"HR = {hr:.2f} [{ci_lower:.2f} – {ci_upper:.2f}]"
    plt.text(
        0.05, 0.12, f"{p_str}\n{hr_str}",
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    )

    plt.title(
        f"{dataset_name} · {endpoint_label}",
        fontsize=13, fontweight="bold",
    )
    plt.xlabel("Time (days)", fontsize=11)
    plt.ylabel("Survival probability", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)

    save_path = os.path.join(
        output_dir, f"km_curve_{dataset_name}.{save_format}"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[KM] {dataset_name} 曲线已保存至: {save_path}")
    print(f"  {p_str} | {hr_str}")

    return {
        "dataset": dataset_name,
        "n_high_risk": n_high,
        "n_low_risk": n_low,
        "median_pi": float(median_pi),
        "p_value": float(p_value),
        "hr": float(hr),
        "hr_ci_lower": float(ci_lower),
        "hr_ci_upper": float(ci_upper),
        "save_path": save_path,
    }


def plot_multi_cohort_km(
    predictions_csvs: List[str],
    dataset_names: List[str],
    endpoint_labels: Optional[List[str]] = None,
    output_dir: str = "outputs/figures",
    save_format: str = "png",
) -> Dict[str, Dict]:
    """
    Plot KM curves for multiple cohorts in a batch (the four-panel Figure 4 in the paper).

    Args:
        predictions_csvs: List of predictions CSV paths for each cohort.
        dataset_names: List of cohort names.
        endpoint_labels: Clinical endpoint labels for each cohort (all default to "OS").
        output_dir: Output directory.
        save_format: Save format.

    Returns:
        all_results: {dataset_name: result_dict}
    """
    if endpoint_labels is None:
        endpoint_labels = ["OS"] * len(dataset_names)

    all_results = {}

    for csv_path, name, endpoint in zip(
        predictions_csvs, dataset_names, endpoint_labels
    ):
        if not os.path.exists(csv_path):
            print(f"[KM] 跳过 {name}: 文件不存在 {csv_path}")
            continue

        result = plot_kaplan_meier(
            predictions_csv=csv_path,
            dataset_name=name,
            endpoint_label=endpoint,
            output_dir=output_dir,
            save_format=save_format,
        )
        all_results[name] = result

    return all_results


# ==============================================================================
# Low-level utility functions (used by other modules).
# ==============================================================================

def split_risk_groups(
    pi_scores: np.ndarray,
    method: str = "median",
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Split high- and low-risk groups by PI (Section 4.6 of the paper).

    X_i = 1 if PI_i >= Median(PI)  (high risk)
    X_i = 0 if PI_i <  Median(PI)  (low risk)
    """
    if threshold is not None:
        cutoff = threshold
    elif method == "median":
        cutoff = np.median(pi_scores)
    elif method == "mean":
        cutoff = np.mean(pi_scores)
    elif method == "quantile":
        cutoff = np.percentile(pi_scores, 66.7)
    else:
        cutoff = np.median(pi_scores)

    return (pi_scores >= cutoff).astype(int)


def compute_log_rank_test(
    times: np.ndarray,
    events: np.ndarray,
    risk_groups: np.ndarray,
) -> Dict:
    """Perform a two-sided log-rank test."""
    high_mask = risk_groups == 1
    low_mask = risk_groups == 0

    result = logrank_test(
        times[high_mask], times[low_mask],
        events[high_mask], events[low_mask],
    )

    return {
        "test_statistic": float(result.test_statistic),
        "p_value": float(result.p_value),
    }


def compute_hazard_ratio(
    times: np.ndarray,
    events: np.ndarray,
    risk_groups: np.ndarray,
) -> Dict:
    """Compute HR and 95% CI with a univariate Cox PH model."""
    df = pd.DataFrame({
        "time": times,
        "event": events,
        "risk_group": risk_groups,
    })

    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col="time", event_col="event", show_progress=False)
        summary = cph.summary
        hr = float(np.exp(summary.loc["risk_group", "coef"]))
        hr_lower = float(np.exp(summary.loc["risk_group", "coef lower 95%"]))
        hr_upper = float(np.exp(summary.loc["risk_group", "coef upper 95%"]))
        coef = float(summary.loc["risk_group", "coef"])
    except Exception:
        hr, hr_lower, hr_upper, coef = 1.0, 1.0, 1.0, 0.0

    return {"hr": hr, "hr_lower": hr_lower, "hr_upper": hr_upper, "coef": coef}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuDAG-Pro KM Survival Analysis")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions CSV 路径")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称")
    parser.add_argument("--endpoint", type=str, default="OS",
                        help="临床终点 (OS/DMFS)")
    parser.add_argument("--output_dir", type=str, default="outputs/figures")
    parser.add_argument("--format", type=str, default="png")

    args = parser.parse_args()

    plot_kaplan_meier(
        predictions_csv=args.predictions,
        dataset_name=args.dataset,
        endpoint_label=args.endpoint,
        output_dir=args.output_dir,
        save_format=args.format,
    )
