import numpy as np
import torch

from benchmarl.environments.sar.sarwrapper import SARWrapper


def _preference_sampler(env_type: str, env_index: int = 0) -> SARWrapper:
    wrapper = object.__new__(SARWrapper)
    wrapper.fixed_preference = None
    wrapper.condition_on_preference = True
    wrapper.env_type = env_type
    wrapper.max_episodes = 5
    wrapper.env_index = env_index
    wrapper.u_sample_start = 0.0
    return wrapper


def test_sar_evaluation_preference_is_two_dimensional_grid_point():
    wrapper = _preference_sampler("eval", env_index=2)

    preference = wrapper.u_sample_preference()

    np.testing.assert_allclose(preference, [0.5, 0.5])
    assert preference.shape == (2,)


def test_sar_training_preference_is_on_two_objective_simplex():
    wrapper = _preference_sampler("train")

    preference = wrapper.u_sample_preference()

    assert preference.shape == (2,)
    assert np.all(preference >= 0.0)
    np.testing.assert_allclose(preference.sum(), 1.0)


def test_downsample_keeps_mlagents_normalized_visual_range():
    image = np.full((1, 4, 4, 3), 0.5, dtype=np.float32)

    result = SARWrapper.downsample_image(None, image, out_h=2, out_w=2)

    torch.testing.assert_close(torch.from_numpy(result), torch.full((1, 4), 0.5))


def test_downsample_normalizes_uint8_like_visual_input():
    image = np.full((1, 4, 4, 3), 255, dtype=np.uint8)

    result = SARWrapper.downsample_image(None, image, out_h=2, out_w=2)

    torch.testing.assert_close(torch.from_numpy(result), torch.ones(1, 4))
