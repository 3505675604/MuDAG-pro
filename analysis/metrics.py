"""
MuDAG-Pro advanced evaluation-metrics module (Sections 4.2 and 4.8 of the paper).

Implements the following prognostic evaluation metrics:
- C-index (Harrell's Concordance Index; Section 4.2 of the paper).
- Time-dependent AUC(t) (Section 4.2 of the paper).
- IBS (Integrated Brier Score)
- NRI (Net Reclassification Improvement; Section 4.8 of the paper).
- IDI (Integrated Discrimination Improvement; Section 4.8 of the paper).
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from lifelines.utils import concordance_index


def compute_c_index(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> float:
    """
    Compute Harrell's C-index (Sections 3.7 and 4.2 of the paper).

    C-index = Σ_{(i,j)∈P} [I(PI_i > PI_j) + 0.5·I(PI_i = PI_j)] / |P|
    where P = {(i,j) | T_i < T_j and δ_i = 1}.

    Args:
        risk_scores: Predicted risk scores (PI); larger values indicate higher risk.
        times: Observed survival times.
        events: Event indicators (1=event, 0=censored).

    Returns:
        c_index: Concordance index.
    """
    # lifelines convention: concordance_index(event_times, predicted_scores, event_observed).
    # Higher risk score → shorter survival time, so pass -risk_scores.
    return concordance_index(times, -risk_scores, events)


def compute_c_index_bootstrap(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute the bootstrap mean and standard deviation of the C-index (Section 4.3 of the paper).

    All C-index results are based on 1,000 nonparametric bootstrap resamples.

    Returns:
        mean_c_index: Mean C-index.
        std_c_index: C-index standard deviation.
    """
    rng = np.random.RandomState(seed)
    n = len(times)
    c_indices = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        c_idx = compute_c_index(risk_scores[idx], times[idx], events[idx])
        c_indices.append(c_idx)

    return float(np.mean(c_indices)), float(np.std(c_indices))


def calculate_time_dependent_auc(
    times: np.ndarray,
    events: np.ndarray,
    pi_scores: np.ndarray,
    eval_time: float,
) -> float:
    """
    Compute time-dependent AUC(t) at a specified follow-up time t (Section 4.2 of the paper).

    Uses the cumulative/dynamic (C/D) AUC definition:
    AUC(t) = P(PI_i > PI_j | T_i ≤ t, δ_i = 1 and T_j > t)

    Args:
        times: Observed survival times.
        events: Event indicators.
        pi_scores: Predicted PI scores.
        eval_time: Evaluation time point (days).

    Returns:
        auc_t: Time-dependent AUC value.
    """
    # Cases: T_i ≤ eval_time and δ_i = 1.
    case_mask = (times <= eval_time) & (events == 1)
    # Controls: T_j > eval_time.
    control_mask = (times > eval_time)

    n_cases = np.sum(case_mask)
    n_controls = np.sum(control_mask)

    if n_cases == 0 or n_controls == 0:
        return 0.5

    cases_pi = pi_scores[case_mask]
    controls_pi = pi_scores[control_mask]

    # Compute the conditional probability AUC(t).
    concordant_pairs = 0
    total_pairs = n_cases * n_controls

    for c in cases_pi:
        concordant_pairs += (
            np.sum(c > controls_pi) + 0.5 * np.sum(np.abs(c - controls_pi) < 1e-8)
        )

    auc_t = concordant_pairs / total_pairs
    return auc_t


def compute_time_dependent_auc_curve(
    times: np.ndarray,
    events: np.ndarray,
    pi_scores: np.ndarray,
    eval_times: List[float],
) -> Dict[float, float]:
    """
    Compute AUC(t) at multiple time points (Section 4.2 of the paper).

    Args:
        times: Survival times.
        events: Event indicators.
        pi_scores: PI scores.
        eval_times: List of evaluation times (for example, [365, 1095, 1825, 2555] for 1/3/5/7 years).

    Returns:
        auc_dict: {time: auc_value}
    """
    auc_dict = {}
    for t_eval in eval_times:
        auc_dict[t_eval] = calculate_time_dependent_auc(
            times, events, pi_scores, t_eval
        )
    return auc_dict


def calculate_integrated_brier_score(
    times: np.ndarray,
    events: np.ndarray,
    pi_scores: np.ndarray,
    eval_times: Optional[List[float]] = None,
) -> float:
    """
    Compute the Integrated Brier Score (IBS).

    Brier(t) = 1/N Σ [Ŝ(t|x_i)² · I(T_i ≤ t, δ_i=1) / Ĝ(T_i)
                      + (1-Ŝ(t|x_i))² · I(T_i > t) / Ĝ(t)]

    Here Ĝ is the Kaplan-Meier estimate of the censoring distribution, and Ŝ is the predicted survival probability.

    Args:
        times: Survival times.
        events: Event indicators.
        pi_scores: PI scores.
        eval_times: Evaluation time points (default: 100 equally spaced points from 0 to max(times)).

    Returns:
        ibs: Integrated Brier Score.
    """
    if eval_times is None:
        eval_times = np.linspace(0, times.max() * 0.8, 100).tolist()

    from lifelines import KaplanMeierFitter

    n = len(times)

    # Estimate the censoring survival function Ĝ(t) (Kaplan-Meier on censoring).
    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(times, event_observed=(1 - events))
    g_func = kmf_censor.survival_function_at_times

    # Approximate survival probability with PI: Ŝ(t|x) = exp(-H_0(t) · exp(PI)).
    # Simplification: estimate the baseline cumulative hazard with Nelson-Aalen.
    baseline_hazard = _estimate_baseline_hazard_nelson_aalen(times, events, pi_scores)

    brier_scores = []
    for t_eval in eval_times:
    # Baseline cumulative hazard H_0(t).
        H0_t = baseline_hazard[
            baseline_hazard[:, 0] <= t_eval
        ][:, 1].sum() if len(baseline_hazard) > 0 else 0.0

        surv_pred = np.exp(-H0_t * np.exp(pi_scores))
        surv_pred = np.clip(surv_pred, 0.0, 1.0)

        brier_t = 0.0
        for i in range(n):
            g_val = float(kmf_censor.survival_function_at_times(
                [min(times[i], t_eval)]
            ).values[0])
            if g_val < 1e-6:
                g_val = 1e-6

            if times[i] <= t_eval and events[i] == 1:
                brier_t += (surv_pred[i]) ** 2 / g_val
            elif times[i] > t_eval:
                brier_t += (1.0 - surv_pred[i]) ** 2 / g_val

        brier_scores.append(brier_t / n)

    # Numerical integration.
    ibs = float(np.trapz(brier_scores, eval_times) / (eval_times[-1] - eval_times[0]))
    return ibs


def _estimate_baseline_hazard_nelson_aalen(
    times: np.ndarray,
    events: np.ndarray,
    pi_scores: np.ndarray,
) -> np.ndarray:
    """Estimate the baseline cumulative hazard with Nelson-Aalen (Breslow method)."""
    unique_times = np.unique(times[events == 1])
    if len(unique_times) == 0:
        return np.zeros((0, 2))

    baseline = []
    for t in unique_times:
        at_risk = times >= t
        risk_exp = np.exp(pi_scores[at_risk])
        events_at_t = np.sum((times == t) & (events == 1))
        if np.sum(risk_exp) > 0:
            h0 = events_at_t / np.sum(risk_exp)
        else:
            h0 = 0.0
        baseline.append([t, h0])

    return np.array(baseline)


def calculate_nri_idi(
    p_base: np.ndarray,
    p_new: np.ndarray,
    events: np.ndarray,
) -> Dict:
    """
    Compute the Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI) (Section 4.8 of the paper).

    NRI (Net Reclassification Improvement):
        NRI = (P_up|event - P_down|event) + (P_down|non-event - P_up|non-event)

    IDI (Integrated Discrimination Improvement):
        IDI = (IS_new - IS_old) - (IP_new - IP_old)
        where IS = mean(p|event), IP = mean(p|non-event).

    Args:
        p_base: Predicted probabilities/scores from the baseline model.
        p_new: Predicted probabilities/scores from the new model.
        events: Event indicators (1=event, 0=censored).

    Returns:
        {"NRI": nri_value, "IDI": idi_value}
    """
    event_mask = (events == 1)
    nonevent_mask = (events == 0)

    n_events = np.sum(event_mask)
    n_nonevents = np.sum(nonevent_mask)

    if n_events == 0 or n_nonevents == 0:
        return {"NRI": 0.0, "IDI": 0.0}

    # ================================================================
    # 1. Compute NRI (Section 4.8 of the paper).
    # ================================================================
    # Event group: upward reclassification (p_new > p_base) is positive.
    events_up = np.sum(p_new[event_mask] > p_base[event_mask])
    events_down = np.sum(p_new[event_mask] < p_base[event_mask])

    # Non-event group: downward reclassification (p_new < p_base) is positive.
    nonevents_up = np.sum(p_new[nonevent_mask] > p_base[nonevent_mask])
    nonevents_down = np.sum(p_new[nonevent_mask] < p_base[nonevent_mask])

    nri = (
        (events_up - events_down) / n_events +
        (nonevents_down - nonevents_up) / n_nonevents
    )

    # ================================================================
    # 2. Compute IDI (Section 4.8 of the paper).
    # ================================================================
    # IS (Integrated Sensitivity): mean predicted probability in the event group.
    is_new = np.mean(p_new[event_mask])
    is_old = np.mean(p_base[event_mask])

    # IP (Integrated 1-Specificity): mean predicted probability in the non-event group.
    ip_new = np.mean(p_new[nonevent_mask])
    ip_old = np.mean(p_base[nonevent_mask])

    idi = (is_new - is_old) - (ip_new - ip_old)

    return {"NRI": float(nri), "IDI": float(idi)}


def compute_all_metrics(
    pi_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    eval_times: Optional[List[float]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Compute all evaluation metrics at once (convenience function).

    Returns:
        {
            "c_index": ...,
            "c_index_bootstrap_mean": ...,
            "c_index_bootstrap_std": ...,
            "auc": {time: auc},
            "ibs": ...,
        }
    """
    if eval_times is None:
        eval_times = [365, 1095, 1825, 2555]  # 1, 3, 5, and 7 years.

    c_idx = compute_c_index(pi_scores, times, events)
    c_mean, c_std = compute_c_index_bootstrap(
        pi_scores, times, events, n_bootstrap, seed
    )
    auc_dict = compute_time_dependent_auc_curve(
        times, events, pi_scores, eval_times
    )
    ibs = calculate_integrated_brier_score(times, events, pi_scores)

    return {
        "c_index": float(c_idx),
        "c_index_bootstrap_mean": float(c_mean),
        "c_index_bootstrap_std": float(c_std),
        "auc": {str(k): float(v) for k, v in auc_dict.items()},
        "ibs": float(ibs),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuDAG-Pro Metrics Calculation")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions CSV 路径")
    parser.add_argument("--eval_times", type=int, nargs="+",
                        default=[365, 1095, 1825, 2555],
                        help="AUC 评估时间点 (天)")

    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    pi = df["predicted_pi"].values
    times = df["time"].values
    events = df["event"].values

    results = compute_all_metrics(
        pi_scores=pi,
        times=times,
        events=events,
        eval_times=args.eval_times,
    )

    print("=" * 50)
    print("Evaluation Metrics")
    print("=" * 50)
    print(f"  C-index:      {results['c_index']:.4f} "
          f"(± {results['c_index_bootstrap_std']:.4f})")
    for t, auc in results["auc"].items():
        print(f"  AUC({int(float(t)/365)}yr):  {auc:.4f}")
    print(f"  IBS:          {results['ibs']:.4f}")
