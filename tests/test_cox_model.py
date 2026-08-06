import math

import pytest
import torch

from src.cox_model import RidgeCoxSurvivalModel, train_ridge_cox


def test_cox_loss_uses_the_full_risk_set_for_tied_event_times():
    model = RidgeCoxSurvivalModel(in_features=1, l2_reg=0.0)
    times = torch.tensor([10.0, 10.0, 5.0])
    events = torch.tensor([1.0, 1.0, 1.0])
    risks = torch.tensor([[0.0], [1.0], [2.0]])

    loss = model.compute_loss(risks, times, events)
    permuted_loss = model.compute_loss(
        risks[[1, 0, 2]], times[[1, 0, 2]], events[[1, 0, 2]]
    )

    tied_log_risk = math.log(math.exp(0.0) + math.exp(1.0))
    final_log_risk = math.log(
        math.exp(0.0) + math.exp(1.0) + math.exp(2.0)
    )
    expected = -(
        (0.0 - tied_log_risk)
        + (1.0 - tied_log_risk)
        + (2.0 - final_log_risk)
    )
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert permuted_loss.item() == pytest.approx(expected, rel=1e-6)


def test_train_ridge_cox_early_stops_when_full_batch_loss_is_constant():
    model = RidgeCoxSurvivalModel(in_features=2, l2_reg=1.0)
    features = torch.zeros((4, 2))
    times = torch.tensor([1.0, 2.0, 3.0, 4.0])
    events = torch.ones(4)

    history = train_ridge_cox(
        model,
        features,
        times,
        events,
        learning_rate=0.01,
        max_epochs=50,
        patience=3,
        verbose=False,
    )

    assert len(history["train_loss"]) == 4
