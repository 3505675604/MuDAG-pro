"""
MuDAG-Pro decision curve analysis module (DCA, net benefit), reproducing Figure 6 and Section 4.7 of the paper.

Evaluates the model's net benefit at different risk thresholds
to help assess its potential decision utility in risk-stratification scenarios.

Equation from Section 4.7 of the paper:
    NB(pt) = TP/N - (FP/N) · pt/(1-pt)

DCA compares the model's net benefit with the default Treat All and Treat None strategies.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List


def calculate_net_benefit(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Compute decision net benefit at specified risk thresholds (Section 4.7 of the paper).

    Formula: NB(pt) = (TP / N) - (FP / N) · (pt / (1 - pt))

    Where:
        TP: Number of patients classified as high risk who actually experienced an event.
        FP: Number of patients classified as high risk who did not experience an event.
        pt: Risk threshold.
        N: Total number of patients.
        pt/(1-pt): Cost weight of a false positive relative to a true positive.

    Args:
        y_true: Binary event labels (1=event, 0=no event).
        y_pred_prob: Predicted risk probabilities (continuous values in [0, 1]).
        thresholds: Sequence of risk thresholds.

    Returns:
        net_benefits: Vector of net benefits at each threshold.
    """
    n = len(y_true)
    net_benefits = []

    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0.0)
            continue

        # Mask for samples classified as high risk.
        pred_high = (y_pred_prob >= pt)
        tp = np.sum((pred_high == 1) & (y_true == 1))
        fp = np.sum((pred_high == 1) & (y_true == 0))

        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)


def plot_dca_curve(
    predictions_csv: str,
    dataset_name: str,
    output_dir: str = "outputs/figures",
    save_format: str = "png",
    eval_year: int = 5,
) -> Dict:
    """
    Plot decision curves (Decision Curve Analysis, DCA; Section 4.7 and Figure 6 of the paper).

    Args:
        predictions_csv: Path to the predictions CSV produced by evaluation.
            Must contain the columns predicted_pi, time, and event.
        dataset_name: Dataset name.
        output_dir: Chart output directory.
        save_format: Save format.
        eval_year: Evaluation time point in years (default 5 years = 1825 days).

    Returns:
        results: Dictionary containing net benefits at each threshold.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(predictions_csv)

    # Compute 5-year event status (binarized).
    eval_time_days = eval_year * 365
    y_true = ((df["time"] <= eval_time_days) & (df["event"] == 1)).astype(int).values

    # Map the continuous prognostic index (PI) to probability form in [0, 1].
    pi_scores = df["predicted_pi"].values
    # Normalize PI to [0, 1] with a sigmoid.
    pred_probs = 1.0 / (1.0 + np.exp(-pi_scores))

    # Threshold range: 1%–50% (the paper focuses on the 10%–30% interval).
    thresholds = np.linspace(0.01, 0.50, 100)

    # ================================================================
    # 1. Net benefit of the MuDAG-Pro model.
    # ================================================================
    model_nb = calculate_net_benefit(y_true, pred_probs, thresholds)

    # ================================================================
    # 2. Net benefit of the Treat All strategy (classify all patients as high risk).
    # ================================================================
    treat_all_nb = calculate_net_benefit(
        y_true, np.ones_like(pred_probs), thresholds
    )

    # ================================================================
    # 3. Net benefit of the Treat None strategy is always 0.
    # ================================================================
    treat_none_nb = np.zeros_like(thresholds)

    # ================================================================
    # 4. Plot the curves (Figure 6 of the paper).
    # ================================================================
    plt.figure(figsize=(8, 6))

    # MuDAG-Pro curve.
    plt.plot(
        thresholds * 100, model_nb,
        label="MuDAG-Pro", color="#1f77b4", linewidth=2.5,
    )
    # Treat All curve.
    plt.plot(
        thresholds * 100, treat_all_nb,
        label="Treat All", color="#7f7f7f", linestyle=":",
    )
    # Treat None curve.
    plt.plot(
        thresholds * 100, treat_none_nb,
        label="Treat None", color="#2ca02c", linestyle="-",
    )

    # Highlight the key 10%–30% diagnostic-threshold interval (Section 4.7 of the paper).
    plt.axvspan(
        10, 30, color="#1f77b4", alpha=0.1,
        label="Relevant Range (10%–30%)",
    )

    plt.title(
        f"Decision Curve Analysis ({eval_year}-Year Risk) — {dataset_name}",
        fontsize=13, fontweight="bold",
    )
    plt.xlabel("Threshold probability (%)", fontsize=11)
    plt.ylabel("Net benefit", fontsize=11)
    plt.ylim(-0.05, max(model_nb.max(), 0.15) + 0.02)
    plt.xlim(0, 50)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)

    save_path = os.path.join(
        output_dir, f"dca_curve_{dataset_name}.{save_format}"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Compute the mean net benefit over the 10%–30% range.
    focus_mask = (thresholds >= 0.10) & (thresholds <= 0.30)
    avg_nb_focus = float(model_nb[focus_mask].mean())

    print(f"[DCA] {dataset_name} 决策曲线已保存至: {save_path}")
    print(f"  平均净收益 (10%–30%): {avg_nb_focus:.4f}")

    return {
        "dataset": dataset_name,
        "eval_year": eval_year,
        "thresholds": thresholds.tolist(),
        "model_nb": model_nb.tolist(),
        "treat_all_nb": treat_all_nb.tolist(),
        "avg_nb_focus_range": avg_nb_focus,
        "save_path": save_path,
    }


def plot_multi_model_dca(
    predictions_dict: Dict[str, str],
    dataset_name: str,
    output_dir: str = "outputs/figures",
    save_format: str = "png",
    eval_year: int = 5,
) -> Dict:
    """
    Compare DCA results for multiple models (Section 4.7 of the paper).

    Args:
        predictions_dict: {"Clinical Cox": "path/to/clinical_predictions.csv",
                           "Pathway Cox": "path/to/pathway_predictions.csv",
                           "MuDAG-Pro": "path/to/mudag_predictions.csv"}
        dataset_name: Dataset name.
        output_dir: Output directory.
        save_format: Save format.
        eval_year: Evaluation time point.

    Returns:
        results: DCA results for each model.
    """
    os.makedirs(output_dir, exist_ok=True)
    eval_time_days = eval_year * 365

    # Use data from the first model to obtain y_true.
    first_model = list(predictions_dict.keys())[0]
    df_ref = pd.read_csv(predictions_dict[first_model])
    y_true = ((df_ref["time"] <= eval_time_days) & (df_ref["event"] == 1)).astype(int).values

    thresholds = np.linspace(0.01, 0.50, 100)

    plt.figure(figsize=(10, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    all_results = {}

    for i, (model_name, csv_path) in enumerate(predictions_dict.items()):
        if not os.path.exists(csv_path):
            print(f"[DCA] 跳过 {model_name}: 文件不存在")
            continue

        df = pd.read_csv(csv_path)
        pi_scores = df["predicted_pi"].values
        pred_probs = 1.0 / (1.0 + np.exp(-pi_scores))

        model_nb = calculate_net_benefit(y_true, pred_probs, thresholds)
        color = colors[i % len(colors)]

        plt.plot(
            thresholds * 100, model_nb,
            label=model_name, color=color, linewidth=2.0,
        )
        all_results[model_name] = {
            "thresholds": thresholds.tolist(),
            "net_benefit": model_nb.tolist(),
        }

    # Treat All / Treat None
    treat_all_nb = calculate_net_benefit(
        y_true, np.ones(len(y_true)), thresholds
    )
    plt.plot(
        thresholds * 100, treat_all_nb,
        label="Treat All", color="gray", linestyle="--",
    )
    plt.plot(
        thresholds * 100, np.zeros_like(thresholds),
        label="Treat None", color="green", linestyle=":",
    )

    plt.axvspan(10, 30, color="blue", alpha=0.08, label="Focus (10%–30%)")

    plt.title(
        f"Decision Curve Analysis — {dataset_name} ({eval_year}-Year)",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel("Threshold probability (%)", fontsize=12)
    plt.ylabel("Net benefit", fontsize=12)
    plt.xlim(0, 50)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9, frameon=True)

    save_path = os.path.join(
        output_dir, f"dca_multi_model_{dataset_name}.{save_format}"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DCA] 多模型比较图已保存至: {save_path}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuDAG-Pro DCA Analysis")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions CSV 路径")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="outputs/figures")
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--eval_year", type=int, default=5,
                        help="评估时间点 (年)")

    args = parser.parse_args()

    plot_dca_curve(
        predictions_csv=args.predictions,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        save_format=args.format,
        eval_year=args.eval_year,
    )
