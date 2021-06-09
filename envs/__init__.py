"""Register custom environment in OpenAI Gym."""

import gym.envs.registration as gym_registration
import numpy as np


# Humanoid envs
gym_registration.register(
    id='HumanoidOneDestination-v0',
    entry_point='envs.humanoid_hide_and_seek:HumanoidEnv',
    max_episode_steps=1000,
    kwargs={'destinations': dict(
        locs=np.array([[1.5, 1.5], [1, 1.5], [0.5, 1.5], [0, 1.5]]),
        radii=[2.0, 1.5, 1.0, 0.5],
        rewards=[2.0, 2.0, 2.0, 2.0],
        solved=[False, False, False, True])},
)

gym_registration.register(
    id='HumanoidTwoDestinations-v0',
    entry_point='envs.humanoid_hide_and_seek:HumanoidEnv',
    max_episode_steps=1000,
    kwargs={'destinations': dict(
        locs=np.array([[0, 2], [1, 2], [-1, 2], [1.5, 2], [-1.5, 2]]),
        radii=[2.0, 1.0, 1.0, 0.5, 0.5],
        rewards=[2.0, 2.0, 2.0, 2.0, 2.0],
        solved=[False, False, False, True, True])},
)