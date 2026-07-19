import numpy as np
import torch
from tensordict import TensorDict

from benchmarl.evaluations.mo_metrics import extract_explore_path_points


def test_extract_points_uses_metrics_from_done_transition():
    metrics = torch.zeros(3, 1, 6)
    metrics[1, 0, 2] = 12.0
    metrics[1, 0, 5] = 95.0
    done = torch.tensor([[False], [True], [False]])
    rollout = TensorDict(
        {"next": {"manual_metrics": metrics, "done": done}},
        batch_size=[3],
    )

    points = extract_explore_path_points([rollout])

    np.testing.assert_allclose(points, [[95.0, -12.0]])
