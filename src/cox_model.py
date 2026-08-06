"""
MuDAG-Pro Ridge-Cox proportional hazards model (Section 3.7 of the paper).

Implements a Cox proportional hazards model with L2 regularization:
- Log partial-likelihood loss + L2 penalty.
- Scalar prognostic index PI_i = β*^T · x̃_i^T.
- Pathway hazard ratio HR_k = exp(β_k) and 95% confidence interval.
- 5-fold cross-validation + grid search for the optimal λ.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union, List


class RidgeCoxSurvivalModel(nn.Module):
    """
    MuDAG-Pro Ridge-Cox proportional hazards model with L2 regularization (Section 3.7 of the paper).

    Accepts the propagation-enhanced 331-dimensional pathway features x̃_i,
    outputs the prognostic index (PI), and optimizes the L2-penalized partial-likelihood loss.

    Mathematical formulas:
        h_i(t) = h_0(t) · exp(x̃_i · β^T)

        L(β) = -Σ_d [β^T·x̃_{i_d} - log(Σ_{j∈R_d} exp(β^T·x̃_j))] + λ||β||²₂

        PI_i = β*^T · x̃_i^T
        HR_k = exp(β_k)
    """
    def __init__(self, in_features: int = 331, l2_reg: float = 1e-3):
        """
        Args:
            in_features: Number of pathway nodes M (default 331).
            l2_reg: L2 regularization hyperparameter λ.
        """
        super().__init__()
        self.in_features = in_features
        self.l2_reg = l2_reg

        # Linear regression coefficients β (without a bias term, consistent with the Cox model definition).
        self.linear = nn.Linear(in_features, 1, bias=False)

        # Initialize weights (with small random values or zeros).
        nn.init.zeros_(self.linear.weight)

    def forward(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar prognostic index PI_i = β^T · x̃_i^T (Equation 3.7 in the paper).

        Args:
            x_tilde: Propagation-enhanced pathway representation matrix X̃ (shape: N x M).

        Returns:
            PI: Scalar prognostic-index vector (shape: N x 1).
        """
        pi = self.linear(x_tilde)
        return pi

    def compute_loss(
        self,
        pi: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the negative log partial-likelihood loss with an L2 penalty (Equation 3.7 in the paper).

        ℓ(β) = Σ_d [β^T·x̃_{i_d} - log Σ_{j∈R_d} exp(β^T·x̃_j)] - λ||β||²₂

        Note: The paper uses a sum (Σ_d), not a mean, so the loss grows with sample size.
        For numerical stability, clamp pi values to prevent exp overflow.

        Args:
            pi: Predicted prognostic-index vector (shape: N x 1 or N).
            times: Survival/observation-time vector T_i (shape: N).
            events: Event-indicator vector δ_i (1=event, 0=right-censored) (shape: N).

        Returns:
            total_loss: Scalar loss value.
        """
        pi = pi.squeeze(-1)

        # Numerical stability: clamp pi to prevent exp overflow.
        pi_clamped = torch.clamp(pi, min=-20.0, max=20.0)

        # 1. Sort by survival time in descending order (an efficient way to handle Cox risk sets R_d).
        sorted_times, order = torch.sort(times, descending=True)
        sorted_events = events[order]
        sorted_pi = pi_clamped[order]

        # 2. Compute log(sum(exp(pi_j))) for each complete risk set. All
        # samples tied at the same observed time share the same denominator;
        # using a running denominator per row would make the loss depend on
        # the arbitrary order of tied samples.
        log_cum_exp_pi = torch.logcumsumexp(sorted_pi, dim=0)
        _, tie_counts = torch.unique_consecutive(
            sorted_times, return_counts=True
        )
        tie_end_indices = torch.cumsum(tie_counts, dim=0) - 1
        tie_log_risk = log_cum_exp_pi[tie_end_indices]
        log_risk_sets = torch.repeat_interleave(tie_log_risk, tie_counts)

        # 3. Compute loss only for samples where an event actually occurred (δ_i = 1).
        event_mask = (sorted_events == 1.0)

        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=pi.device, requires_grad=True)

        # Equation from the paper: sum over each event time point (not the mean).
        partial_log_likelihood = (
            sorted_pi[event_mask] - log_risk_sets[event_mask]
        )
        neg_log_partial_likelihood = -torch.sum(partial_log_likelihood)

        # 4. Add the L2 regularization penalty: λ · ||β||²₂.
        l2_penalty = self.l2_reg * torch.sum(self.linear.weight ** 2)

        total_loss = neg_log_partial_likelihood + l2_penalty
        return total_loss

    def get_prognostic_index(self, x_tilde: torch.Tensor) -> np.ndarray:
        """
        Compute prognostic index PI_i = β*^T · x̃_i^T (Section 3.7 of the paper).

        Args:
            x_tilde: Pathway-feature matrix X̃ in R^(N x M).

        Returns:
            pi: Prognostic-index vector in R^N.
        """
        with torch.no_grad():
            pi = self.forward(x_tilde).squeeze(-1).cpu().numpy()
        return pi

    def get_weights(self) -> np.ndarray:
        """Return regression coefficients β* in R^M."""
        return self.linear.weight.detach().cpu().numpy().squeeze()

    def get_hazard_ratios(self) -> np.ndarray:
        """
        Compute hazard ratios for M=331 pathways: HR_k = exp(β_k*) (Section 3.7 of the paper).

        HR_k > 1: Increased activity of pathway k is positively associated with poor prognosis (risk driver).
        HR_k < 1: Pathway k is a low-risk factor (protective factor).
        """
        weights = self.get_weights()
        hr = np.exp(weights)
        return hr

    def get_hazard_ratios_with_ci(
        self,
        x_tilde: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        alpha: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """
        Compute pathway hazard ratios HR_k and 95% confidence intervals (Section 3.7 of the paper).

        Estimate standard errors from asymptotic normality and the inverse Hessian:
            Var(β*) = [-H_ℓ(β*) + 2λI]^{-1}
            SE(β_k*) = √Var(β*)_kk
            95% CI_k = exp(β_k* ± 1.96 · SE(β_k*))

        Args:
            x_tilde: Pathway-feature matrix.
            times: Survival times.
            events: Event indicators.
            alpha: Significance level (default 0.05 → 95% CI).

        Returns:
            {
                "beta": β vector,
                "hr": HR vector,
                "hr_lower": lower bound of the 95% CI,
                "hr_upper": upper bound of the 95% CI,
                "se": standard-error vector
            }
        """
        # Approximate the Hessian numerically.
        beta = self.linear.weight.clone().detach()
        M = beta.shape[1]

        # Compute a finite-difference approximation of the Hessian matrix.
        eps = 1e-4
        hessian = torch.zeros((M, M))

        x_tilde_detached = x_tilde.detach()
        times_detached = times.detach()
        events_detached = events.detach()

        # Compute the gradient.
        pi = self.forward(x_tilde_detached)
        loss = self.compute_loss(pi, times_detached, events_detached)

        # Compute the Hessian through automatic differentiation.
        grad = torch.autograd.grad(loss, self.linear.weight, create_graph=True)[0]

        for i in range(M):
            grad_i = grad[0, i]
            hessian_row = torch.autograd.grad(
                grad_i, self.linear.weight, retain_graph=(i < M - 1)
            )[0]
            hessian[i, :] = hessian_row

        # Var(β*) = [-H_ℓ(β*) + 2λI]^{-1}
        hessian_np = hessian.detach().cpu().numpy()
        fisher_info = -hessian_np + 2.0 * self.l2_reg * np.eye(M)

        try:
            var_beta = np.linalg.inv(fisher_info)
            se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
        except np.linalg.LinAlgError:
            # Use the pseudoinverse when the matrix is singular.
            var_beta = np.linalg.pinv(fisher_info)
            se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))

        beta_np = beta.detach().cpu().numpy().squeeze()
        hr = np.exp(beta_np)

        z_score = 1.96  # 95% CI
        hr_lower = np.exp(beta_np - z_score * se)
        hr_upper = np.exp(beta_np + z_score * se)

        return {
            "beta": beta_np,
            "hr": hr,
            "hr_lower": hr_lower,
            "hr_upper": hr_upper,
            "se": se,
        }

    def get_top_pathways(
        self,
        pathway_names: List[str],
        top_k: int = 20,
        x_tilde: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None,
        events: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """
        Return the top K key prognostic driver pathways with the largest |β_k| (Section 3.7 of the paper).

        Args:
            pathway_names: List of pathway names.
            top_k: Number of top pathways to return.
            x_tilde: Feature matrix (optional, used to compute CIs).
            times: Survival times (optional).
            events: Event indicators (optional).

        Returns:
            Sorted list of pathways, each containing the pathway name, β, HR, and 95% CI.
        """
        beta = self.get_weights()

        # If data is provided, compute HRs with CIs.
        if x_tilde is not None and times is not None and events is not None:
            hr_info = self.get_hazard_ratios_with_ci(x_tilde, times, events)
            hr = hr_info["hr"]
            hr_lower = hr_info["hr_lower"]
            hr_upper = hr_info["hr_upper"]
        else:
            hr = np.exp(beta)
            hr_lower = hr.copy()
            hr_upper = hr.copy()

        # Sort by |β| in descending order.
        sorted_idx = np.argsort(np.abs(beta))[::-1][:top_k]

        top_pathways = []
        for idx in sorted_idx:
            top_pathways.append({
                "rank": len(top_pathways) + 1,
                "pathway": pathway_names[idx] if idx < len(pathway_names) else f"Pathway_{idx}",
                "beta": float(beta[idx]),
                "hr": float(hr[idx]),
                "hr_95ci_lower": float(hr_lower[idx]),
                "hr_95ci_upper": float(hr_upper[idx]),
                "risk_direction": "risk" if beta[idx] > 0 else "protective",
            })

        return top_pathways


# ==============================================================================
# Training and evaluation utility functions.
# ==============================================================================

def train_ridge_cox(
    model: RidgeCoxSurvivalModel,
    x_train: torch.Tensor,
    times_train: torch.Tensor,
    events_train: torch.Tensor,
    x_val: Optional[torch.Tensor] = None,
    times_val: Optional[torch.Tensor] = None,
    events_val: Optional[torch.Tensor] = None,
    learning_rate: float = 0.001,
    max_epochs: int = 500,
    patience: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Train a Ridge-Cox model.

    Args:
        model: RidgeCoxSurvivalModel instance.
        x_train: Training-set pathway features X̃ in R^(N_train x M).
        times_train: Training-set survival times.
        events_train: Training-set event indicators.
        x_val: Validation-set features (optional).
        times_val: Validation-set survival times (optional).
        events_val: Validation-set event indicators (optional).
        learning_rate: Learning rate.
        max_epochs: Maximum number of training epochs.
        patience: Early-stopping patience.
        verbose: Whether to print training information.

    Returns:
        history: Training history.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    history = {"train_loss": [], "val_loss": [], "nll_loss": []}
    best_val_loss = float('inf')
    best_weights = model.linear.weight.detach().clone()
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        pi = model(x_train)
        total_loss = model.compute_loss(pi, times_train, events_train)

        # Compute the unpenalized negative log-likelihood (for monitoring).
        with torch.no_grad():
            l2_penalty = model.l2_reg * torch.sum(model.linear.weight ** 2)
            nll_loss = total_loss - l2_penalty

        total_loss.backward()
        optimizer.step()

        train_loss_value = total_loss.detach().item()
        history["train_loss"].append(train_loss_value)
        history["nll_loss"].append(nll_loss.detach().item())

        # Validate.
        if x_val is not None and times_val is not None and events_val is not None:
            model.eval()
            with torch.no_grad():
                pi_val = model(x_val)
                val_loss = model.compute_loss(pi_val, times_val, events_val)
            val_loss_val = val_loss.detach().item()
            history["val_loss"].append(val_loss_val)
            scheduler.step(val_loss)
            monitored_loss = val_loss_val
            monitor_name = "validation"
        else:
            scheduler.step(total_loss)
            monitored_loss = train_loss_value
            monitor_name = "training"

        if monitored_loss < best_val_loss - 1e-8:
            best_val_loss = monitored_loss
            best_weights = model.linear.weight.detach().clone()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"  Early stopping at epoch {epoch + 1}, "
                    f"best {monitor_name} loss: {best_val_loss:.6f}"
                )
            break

        if verbose and (epoch + 1) % 50 == 0:
            l2_val = total_loss.detach().item() - nll_loss.detach().item()
            print(f"  Epoch {epoch + 1}/{max_epochs}, "
                  f"Total: {total_loss.detach().item():.6f}, "
                  f"NLL: {nll_loss.detach().item():.6f}, "
                  f"L2: {l2_val:.6f}")

    # Restore the best weights.
    model.linear.weight.data.copy_(best_weights)

    return history


def compute_c_index_fast(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> float:
    """
    Quickly compute Harrell's C-index (Sections 3.7 and 4.2 of the paper).

    C-index = Σ_{(i,j)∈P} [I(PI_i > PI_j) + 0.5·I(PI_i = PI_j)] / |P|
    where P = {(i,j) | T_i < T_j and δ_i = 1}.

    Args:
        risk_scores: Predicted risk scores (larger values indicate higher risk).
        times: Observed survival times.
        events: Event indicators (1=event, 0=censored).

    Returns:
        c_index: Concordance index.
    """
    n = len(times)
    if n < 2:
        return 0.5

    concordant = 0.0
    comparable = 0

    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[i] < times[j]:
                comparable += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif abs(risk_scores[i] - risk_scores[j]) < 1e-8:
                    concordant += 0.5

    if comparable == 0:
        return 0.5
    return concordant / comparable


def grid_search_lambda(
    x_train: torch.Tensor,
    times_train: torch.Tensor,
    events_train: torch.Tensor,
    lambda_candidates: List[float],
    n_folds: int = 5,
    learning_rate: float = 0.001,
    max_epochs: int = 500,
    patience: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[float, float, Dict]:
    """
    Grid-search for the optimal L2 regularization parameter λ (Section 4.1 of the paper).

    Perform 5-fold cross-validation on the TCGA-BRCA training set of 694 cases
    and select the λ value with the highest C-index.

    Args:
        x_train: Training-set pathway features.
        times_train: Training-set survival times.
        events_train: Training-set event indicators.
        lambda_candidates: List of candidate λ values.
        n_folds: Number of cross-validation folds.
        learning_rate: Learning rate.
        max_epochs: Maximum number of training epochs.
        patience: Early-stopping patience.
        seed: Random seed.
        verbose: Whether to print information.

    Returns:
        best_lambda: Optimal λ.
        best_cv_score: Best cross-validation C-index.
        cv_results: Cross-validation results for each λ.
    """
    from sklearn.model_selection import KFold

    n_samples = x_train.shape[0]
    n_features = x_train.shape[1]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = {}

    if verbose:
        print(f"Grid search λ over {len(lambda_candidates)} candidates "
              f"with {n_folds}-fold CV...")

    for lam in lambda_candidates:
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
            x_fold_train = x_train[train_idx]
            t_fold_train = times_train[train_idx]
            e_fold_train = events_train[train_idx]
            x_fold_val = x_train[val_idx]
            t_fold_val = times_train[val_idx]
            e_fold_val = events_train[val_idx]

            model = RidgeCoxSurvivalModel(n_features=n_features, l2_reg=lam)
            train_ridge_cox(
                model,
                x_fold_train, t_fold_train, e_fold_train,
                x_fold_val, t_fold_val, e_fold_val,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                verbose=False,
            )

            pi_val = model.get_prognostic_index(x_fold_val)
            c_idx = compute_c_index_fast(
                pi_val, t_fold_val.cpu().numpy(), e_fold_val.cpu().numpy()
            )
            fold_scores.append(c_idx)

        mean_cv = float(np.mean(fold_scores))
        std_cv = float(np.std(fold_scores))
        cv_results[lam] = {"mean_c_index": mean_cv, "std_c_index": std_cv}

        if verbose:
            print(f"  λ={lam:.4f}: C-index = {mean_cv:.4f} ± {std_cv:.4f}")

    # Select the optimal λ (highest mean C-index).
    best_lambda = max(cv_results, key=lambda k: cv_results[k]["mean_c_index"])
    best_cv_score = cv_results[best_lambda]["mean_c_index"]

    if verbose:
        print(f"Best λ = {best_lambda:.4f} (CV C-index = {best_cv_score:.4f})")

    return best_lambda, best_cv_score, cv_results
