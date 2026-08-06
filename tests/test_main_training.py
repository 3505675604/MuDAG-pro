import main


def test_resolve_training_hyperparameters_reads_config_and_allows_no_propagation():
    config = {
        "propagation": {"alpha": 0.15, "alpha_grid": [0.05, 0.15]},
        "cox": {
            "lambda_range": [0.1, 1.0, 10.0],
            "n_folds": 4,
            "learning_rate": 0.002,
            "max_epochs": 120,
            "early_stopping_patience": 12,
            "batch_size": 16,
        },
    }

    options = main.resolve_training_hyperparameters(config)

    assert options == {
        "model_type": "ridge_cox",
        "alpha_grid": [0.0, 0.05, 0.15],
        "l2_reg_grid": [0.1, 1.0, 10.0],
        "penalizer_grid": [0.001, 0.01, 0.05, 0.1, 0.5],
        "l1_ratio_grid": [0.1, 0.5, 0.9],
        "n_folds": 4,
        "cv_repeats": 3,
        "min_selection_frequency": 0.6,
        "learning_rate": 0.002,
        "max_epochs": 120,
        "patience": 12,
        "batch_size": 16,
    }
