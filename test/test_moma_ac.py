import pytest
import torch
from tensordict import TensorDict

from benchmarl.algorithms.moma_ac import (
    CentralisedVectorCritic,
    MomaACLoss,
    MultiHeadedActor,
)


def _make_loss(policy_delay=2, device="cpu"):
    actor = MultiHeadedActor(4, 2, 2, 2, [16, 16], device)
    critic_1 = CentralisedVectorCritic(8, 4, 2, [16, 16], device)
    critic_2 = CentralisedVectorCritic(8, 4, 2, [16, 16], device)
    return MomaACLoss(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        group="agents",
        n_agents=2,
        n_objectives=2,
        gamma=0.99,
        polyak_tau=0.005,
        policy_delay=policy_delay,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        loss_function="l2",
        action_low=torch.full((2, 2), -1.0),
        action_high=torch.full((2, 2), 1.0),
    )


def _batch(batch_size=5):
    preference = torch.rand(batch_size, 2, 2)
    preference = preference / preference.sum(-1, keepdim=True)
    next_reward = torch.rand(batch_size, 2, 2)
    return TensorDict(
        {
            "agents": {
                "observation": torch.randn(batch_size, 2, 4),
                "preference": preference,
                "action": torch.rand(batch_size, 2, 2) * 2 - 1,
            },
            "next": {
                "agents": {
                    "observation": torch.randn(batch_size, 2, 4),
                    "preference": preference,
                },
                "reward_vector": next_reward,
                "done": torch.zeros(batch_size, 1, dtype=torch.bool),
                "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
            },
        },
        batch_size=[batch_size],
    )


def test_moma_ac_losses_have_gradients_and_vector_priority():
    loss = _make_loss(policy_delay=1)
    batch = _batch()
    output = loss(batch)
    assert output["loss_actor"].requires_grad
    assert output["loss_critic"].requires_grad
    assert batch["agents", "td_error"].shape == (5, 2, 1)


def test_moma_td3_delays_actor_update():
    loss = _make_loss(policy_delay=2)
    loss(_batch())
    assert not loss.last_actor_update
    loss(_batch())
    assert loss.last_actor_update


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_moma_ac_moves_cpu_action_bounds_to_model_device():
    loss = _make_loss(policy_delay=1, device="cuda:0")

    assert loss.action_low.device.type == "cuda"
    assert loss.action_high.device.type == "cuda"
    assert loss._updates.device.type == "cuda"

    output = loss(_batch().to("cuda:0"))
    assert output.device.type == "cuda"
