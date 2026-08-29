from __future__ import annotations

import numpy as np

from ltb_milp.dataset import collect_root_strong_branching_dataset
from ltb_milp.models import BranchingMLP
from ltb_milp.training import predict_scores, train_branching_model


def test_dataset_and_model_training_smoke() -> None:
    dataset = collect_root_strong_branching_dataset(20, 10, 4, seed=0)
    assert dataset.features.ndim == 2
    assert dataset.features.shape[1] == 6
    assert dataset.targets.shape == (dataset.features.shape[0],)
    assert np.all(np.isfinite(dataset.features))
    assert np.all(np.isfinite(dataset.targets))

    model = BranchingMLP()
    history = train_branching_model(model, dataset, epochs=5, learning_rate=1e-3)
    scores = predict_scores(model, dataset.features[:4])
    assert len(history) == 5
    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))
