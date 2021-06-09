"""The Humanoid environment with a sparse reward for approaching destinations.

Based on the Humanoid-v3 environment from OpenAI Gym.
"""

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 1,
    'distance': 4.0,
    'lookat': np.array((0.0, 0.0, 2.0)),
    'elevation': -20.0,
}


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Implements Humanoid with the destination."""

    def __init__(self,
                 xml_file='humanoid.xml',
                 forward_reward_weight=1.25,
                 ctrl_cost_weight=0.1,
                 contact_cost_weight=5e-7,
                 contact_cost_range=(-np.inf, 10.0),
                 healthy_reward=3.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(1.0, 2.0),
                 reset_noise_scale=1e-2,
                 exclude_current_positions_from_observation=False,
                 destinations=None):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._destinations = destinations or {
            'locs': [np.array((2, 2))],
            'radii': [1.],
            'rewards': [3.],
            'solved': [True],
        }

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def seek_reward(self, position):
        """Reward for being in the zone of influence of target destinations.

        It's aggregated (summed) over all zones.
        """
        values = []
        is_solved = False
        for loc, radious, reward, solved in zip(self._destinations['locs'],
                                                self._destinations['radii'],
                                                self._destinations['rewards'],
                                                self._destinations['solved']):
            distance = np.sqrt(np.dot((position - loc), (position - loc)))
            if distance <= radious:
                values.append(reward)
                is_solved = is_solved or solved
        return np.sum(values), is_solved

    def control_cost(self, action):
        del action
        control_cost = self._ctrl_cost_weight * np.sum(
            np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.sim.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.sim.data.qpos[2] < max_z

        return is_healthy

    @property
    def done(self):
        done = ((not self.is_healthy)
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
        ))

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xy_pos_after = mass_center(self.model, self.sim)

        seek_reward, is_solved = self.seek_reward(xy_pos_after)
        healthy_reward = self.healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        rewards = self._forward_reward_weight * seek_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done

        info = {
            'reward_task': seek_reward,
            'reward_quadctrl': -ctrl_cost,
            'reward_alive': healthy_reward,
            'reward_impact': -contact_cost,

            'is_solved': is_solved,

            'x_position': xy_pos_after[0],
            'y_position': xy_pos_after[1],
            'distance_from_origin': np.linalg.norm(xy_pos_after, ord=2),
        }

        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
