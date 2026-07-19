import torch
from tensordict import TensorDict

from benchmarl.algorithms.moma_ac import (
    CentralisedVectorCritic,
    MultiHeadedActor,
)
from benchmarl.algorithms.moma_gpi_pd import (
    ContinuousMultiAgentGPIPolicy,
    MomaGPIPDLoss,
)


def _policy(device="cpu"):
    actor = MultiHeadedActor(4, 2, 2, 2, [16, 16], device)
    critic_args = dict(
        central_observation_dim=8,
        joint_action_dim=4,
        preference_dim=2,
        hidden_sizes=[16, 16],
        device=device,
    )
    return ContinuousMultiAgentGPIPolicy(
        actor=actor,
        critic_1=CentralisedVectorCritic(**critic_args),
        critic_2=CentralisedVectorCritic(**critic_args),
        weight_support=[[1.0, 0.0], [0.0, 1.0]],
        n_agents=2,
        n_objectives=2,
        action_low=torch.full((2, 2), -2.0),
        action_high=torch.full((2, 2), 3.0),
        include_current_preference=True,
    )


def _loss(device="cpu", policy_delay=1):
    return MomaGPIPDLoss(
        policy=_policy(device),
        group="agents",
        n_agents=2,
        n_objectives=2,
        gamma=0.99,
        polyak_tau=0.01,
        policy_delay=policy_delay,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        loss_function="smooth_l1",
        min_priority=0.01,
        priority_exponent=0.6,
        support_sample_probability=0.0,
    )


def _batch(batch_size=7, device="cpu"):
    preference = torch.rand(batch_size, 2, 2, device=device)
    preference = preference / preference.sum(dim=-1, keepdim=True)
    return TensorDict(
        {
            "agents": {
                "observation": torch.randn(
                    batch_size, 2, 4, device=device
                ),
                "preference": preference,
                "action": torch.rand(
                    batch_size, 2, 2, device=device
                )
                * 5.0
                - 2.0,
            },
            "next": {
                "agents": {
                    "observation": torch.randn(
                        batch_size, 2, 4, device=device
                    ),
                    "preference": preference,
                },
                "reward_vector": torch.randn(
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


def test_continuous_gpi_selects_bounded_joint_action():
    policy = _policy()
    observation = torch.randn(5, 2, 4)
    preference = torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]).expand(
        5, -1, -1
    )

    action, value, policy_index = policy(observation, preference)

    assert action.shape == (5, 2, 2)
    assert value.shape == (5, 2)
    assert policy_index.shape == (5,)
    assert torch.all(action >= -2.0)
    assert torch.all(action <= 3.0)
    # Two support policies plus the current requested preference.
    assert policy_index.min() >= 0 and policy_index.max() < 3


def test_gpi_pd_loss_has_actor_critic_gradients_and_priorities():
    loss = _loss()
    batch = _batch()

    output = loss(batch)

    assert output["loss_actor"].requires_grad
    assert output["loss_critic"].requires_grad
    assert batch["agents", "td_error"].shape == (7, 2, 1)
    output["loss_actor"].backward()
    assert any(
        parameter.grad is not None
        for parameter in loss.policy.actor.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in loss.online_critic_parameters
    )
    output["loss_critic"].backward()
    assert all(
        any(parameter.grad is not None for parameter in critic.parameters())
        for critic in (loss.policy.critic_1, loss.policy.critic_2)
    )


def test_gpi_pd_delays_actor_update():
    loss = _loss(policy_delay=2)

    first = loss(_batch())
    assert not loss.last_actor_update
    assert not first["loss_actor"].requires_grad

    second = loss(_batch())
    assert loss.last_actor_update
    assert second["loss_actor"].requires_grad


def test_weight_support_can_be_replaced_without_rebuilding():
    policy = _policy()
    policy.set_weight_support([[0.5, 0.5]])

    _, _, policy_index = policy(
        torch.randn(3, 2, 4),
        torch.full((3, 2, 2), 0.5),
    )

    assert policy_index.max() < 2

