"""
MuDAG-Pro SHAP attribution analysis module (reproduces Figure 5 and Section 4.5 of the paper).

Trains an XGBoost surrogate model using the prognostic index (PI) output by MuDAG-Pro as the supervised target,
then computes SHAP attribution plots to interpret high- and low-risk drivers.

Section 4.5 of the paper:
    Build an XGBoost model from patient-level clinical features, pathway activation states,
    and functional mutation information, then use SHAP values to evaluate each variable's
    contribution to risk prediction. XGBoost is trained with the PI output by MuDAG-Pro as the supervised target.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def run_shap_analysis(
    feature_matrix_csv: str,
    predictions_csv: str,
    dataset_name: str,
    output_dir: str = "outputs/figures",
    save_format: str = "png",
    max_display: int = 15,
    random_state: int = 42,
) -> Dict:
    """
    Train an XGBoost surrogate model using the PI output by MuDAG-Pro as the target,
    then perform SHAP attribution analysis (Section 4.5 and Figure 5 of the paper).

    Args:
        feature_matrix_csv: Path to the feature-matrix CSV.
            Contains pathway activation scores (x̃_i), binary mutation states, and clinical variables.
        predictions_csv: Path to the predictions CSV produced by evaluation.
            Contains the predicted_pi column.
        dataset_name: Dataset name.
        output_dir: Chart output directory.
        save_format: Save format.
        max_display: Maximum number of features displayed in SHAP plots (default 15).
        random_state: Random seed.

    Returns:
        results: Dictionary containing the SHAP importance ranking.
    """
    import xgboost as xgb
    import shap

    os.makedirs(output_dir, exist_ok=True)

    # Read the feature matrix and PI supervised target.
    features_df = pd.read_csv(feature_matrix_csv, index_col=0)
    preds_df = pd.read_csv(predictions_csv, index_col=0)

    # Ensure sample alignment.
    common_samples = features_df.index.intersection(preds_df.index)
    if len(common_samples) == 0:
        raise ValueError("特征矩阵和预测文件的样本 ID 无交集！")

    X = features_df.loc[common_samples]
    y_pi = preds_df.loc[common_samples, "predicted_pi"]

    print(f"[SHAP] {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")

    # ================================================================
    # 1. Train an XGBoost regression surrogate model (Section 4.5 of the paper).
    # ================================================================
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0,
    )
    xgb_model.fit(X, y_pi)

    train_r2 = xgb_model.score(X, y_pi)
    print(f"  XGBoost R² = {train_r2:.4f}")

    # ================================================================
    # 2. Compute SHAP values.
    # ================================================================
    explainer = shap.TreeExplainer(
        xgb_model,
        feature_perturbation="interventional",
        data=X.iloc[:min(100, len(X))],
    )
    shap_values = explainer(X)

    # ================================================================
    # 3. Plot the SHAP summary beeswarm plot (Figure 5 of the paper).
    # ================================================================
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(
        f"SHAP Value Summary — {dataset_name}",
        fontsize=13, fontweight="bold",
    )

    save_path = os.path.join(
        output_dir, f"shap_summary_{dataset_name}.{save_format}"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] 概要图已保存至: {save_path}")

    # ================================================================
    # 4. Plot the SHAP bar chart (sorted by mean(|SHAP|)).
    # ================================================================
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(
        f"SHAP Feature Importance — {dataset_name}",
        fontsize=13, fontweight="bold",
    )

    bar_path = os.path.join(
        output_dir, f"shap_bar_{dataset_name}.{save_format}"
    )
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ================================================================
    # 5. Extract the SHAP importance ranking.
    # ================================================================
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X.columns.tolist(),
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": shap_values.values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    # Print the top 15.
    print(f"\n  Top 15 特征 (按 |SHAP|):")
    for _, row in feature_importance.head(15).iterrows():
        direction = "↑ risk" if row["mean_shap"] > 0 else "↓ protective"
        print(f"    {row['feature']:<30s} "
              f"|SHAP|={row['mean_abs_shap']:.4f} ({direction})")

    return {
        "dataset": dataset_name,
        "train_r2": float(train_r2),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_importance": feature_importance,
        "shap_summary_path": save_path,
        "shap_bar_path": bar_path,
    }


def run_shap_cross_cohort(
    feature_csvs: List[str],
    prediction_csvs: List[str],
    dataset_names: List[str],
    common_features: Optional[List[str]] = None,
    output_dir: str = "outputs/figures",
    save_format: str = "png",
) -> Dict:
    """
    Cross-cohort SHAP consistency analysis (Section 4.5 of the paper).

    Run SHAP analysis separately on multiple cohorts and compare consistency of feature-importance rankings.

    Args:
        feature_csvs: List of feature-matrix CSV paths for each cohort.
        prediction_csvs: List of prediction CSV paths for each cohort.
        dataset_names: List of cohort names.
        common_features: List of common features (optional, used for alignment).
        output_dir: Output directory.
        save_format: Save format.

    Returns:
        all_results: {dataset_name: shap_result}.
    """
    all_results = {}

    for feat_csv, pred_csv, name in zip(
        feature_csvs, prediction_csvs, dataset_names
    ):
        if not os.path.exists(feat_csv) or not os.path.exists(pred_csv):
            print(f"[SHAP] 跳过 {name}: 文件不存在")
            continue

        result = run_shap_analysis(
            feature_matrix_csv=feat_csv,
            predictions_csv=pred_csv,
            dataset_name=name,
            output_dir=output_dir,
            save_format=save_format,
        )
        all_results[name] = result

    # Analyze consistency of top features across cohorts.
    if len(all_results) >= 2:
        names = list(all_results.keys())
        top1 = set(
            all_results[names[0]]["feature_importance"]
            .head(15)["feature"].tolist()
        )
        top2 = set(
            all_results[names[1]]["feature_importance"]
            .head(15)["feature"].tolist()
        )
        overlap = top1 & top2
        print(f"\n[SHAP] 跨队列一致性: {len(overlap)}/{15} "
              f"Top 特征在 {names[0]} 和 {names[1]} 中一致")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuDAG-Pro SHAP Analysis")
    parser.add_argument("--features", type=str, required=True,
                        help="特征矩阵 CSV 路径")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions CSV 路径")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="outputs/figures")
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--max_display", type=int, default=15)

    args = parser.parse_args()

    run_shap_analysis(
        feature_matrix_csv=args.features,
        predictions_csv=args.predictions,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        save_format=args.format,
        max_display=args.max_display,
    )
