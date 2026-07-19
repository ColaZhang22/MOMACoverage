import pytest
import torch
from tensordict import TensorDict

from benchmarl.algorithms.moma_ac import CentralisedVectorCritic
from benchmarl.algorithms.moma_sac import (
    MomaSACLoss,
    MultiHeadedGaussianActor,
)


def _make_loss(device="cpu", fixed_alpha=False):
    actor = MultiHeadedGaussianActor(
        observation_dim=4,
        preference_dim=2,
        action_dim=2,
        n_agents=2,
        hidden_sizes=[16, 16],
        log_std_min=-5.0,
        log_std_max=2.0,
        device=device,
    )
    critic_1 = CentralisedVectorCritic(8, 4, 2, [16, 16], device)
    critic_2 = CentralisedVectorCritic(8, 4, 2, [16, 16], device)
    return MomaSACLoss(
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        group="agents",
        n_agents=2,
        n_objectives=2,
        action_dim=2,
        gamma=0.99,
        polyak_tau=0.005,
        loss_function="smooth_l1",
        action_low=torch.full((2, 2), -1.0),
        action_high=torch.full((2, 2), 1.0),
        alpha_init=0.2,
        target_entropy="auto",
        fixed_alpha=fixed_alpha,
        min_alpha=1e-4,
        max_alpha=10.0,
    )


def _batch(batch_size=5, device="cpu"):
    preference = torch.rand(batch_size, 2, 2, device=device)
    preference = preference / preference.sum(-1, keepdim=True)
    return TensorDict(
        {
            "agents": {
                "observation": torch.randn(batch_size, 2, 4, device=device),
                "preference": preference,
                "action": torch.rand(batch_size, 2, 2, device=device) * 2 - 1,
            },
            "next": {
                "agents": {
                    "observation": torch.randn(
                        batch_size, 2, 4, device=device
                    ),
                    "preference": preference,
                },
                "reward_vector": torch.rand(
                    batch_size, 2, 2, device=device
                ),
                "done": torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=device
                ),
                "terminated": torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=device
                ),
            },
        },
        batch_size=[batch_size],
        device=device,
    )


def test_moma_sac_losses_have_gradients_and_metrics():
    loss = _make_loss()
    batch = _batch()

    output = loss(batch)

    assert output["loss_actor"].requires_grad
    assert output["loss_critic"].requires_grad
    assert output["loss_alpha"].requires_grad
    assert output["alpha"].item() == pytest.approx(0.2)
    assert torch.isfinite(output["entropy"])
    assert batch["agents", "td_error"].shape == (5, 2, 1)


def test_moma_sac_actor_and_critic_optimise_independently():
    loss = _make_loss()
    batch = _batch()
    output = loss(batch)

    output["loss_actor"].backward()
    assert any(parameter.grad is not None for parameter in loss.actor.parameters())
    assert all(
        parameter.grad is None
        for parameter in loss.online_critic_parameters
    )

    output["loss_critic"].backward()
    assert any(
        parameter.grad is not None
        for parameter in loss.online_critic_parameters
    )


def test_moma_sac_fixed_alpha_has_no_alpha_gradient():
    loss = _make_loss(fixed_alpha=True)
    output = loss(_batch())

    assert not output["loss_alpha"].requires_grad
    assert output["loss_alpha"].item() == 0.0


def test_moma_sac_supports_benchmarl_sequential_optimizer_order():
    loss = _make_loss()
    optimizers = (
        (torch.optim.Adam(loss.actor.parameters()), "loss_actor"),
        (
            torch.optim.Adam(list(loss.online_critic_parameters)),
            "loss_critic",
        ),
        (torch.optim.Adam([loss.log_alpha]), "loss_alpha"),
    )
    output = loss(_batch())

    for optimizer, key in optimizers:
        output[key].backward()
        optimizer.step()
        optimizer.zero_grad()

    # The next forward applies the pending target update after the preceding
    # online-network optimiser steps.
    next_output = loss(_batch())
    assert torch.isfinite(next_output["loss_critic"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_moma_sac_moves_bounds_and_alpha_to_cuda():
    loss = _make_loss(device="cuda:0")
    output = loss(_batch(device="cuda:0"))

    assert loss.action_low.device.type == "cuda"
    assert loss.log_alpha.device.type == "cuda"
    assert output.device.type == "cuda"
