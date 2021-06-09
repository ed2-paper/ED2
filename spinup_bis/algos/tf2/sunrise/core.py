"""Core functions of the SUNRISE algorithm."""

import gym
import numpy as np
import tensorflow as tf


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def heuristic_target_entropy(action_space):
    # pylint: disable=line-too-long
    """Copied from https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py"""
    if isinstance(action_space, gym.spaces.Box):  # continuous space
        heuristic_target_entropy = -np.prod(action_space.shape)
    else:
        raise NotImplementedError((type(action_space), action_space))

    return heuristic_target_entropy


def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
        ((value - mu) / (tf.exp(log_std) + EPS)) ** 2 +
        2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu, pi, logp_pi):
    """Applies adjustment to mean, pi and log prob.

    This formula is a little bit magic. To get an understanding of where it
    comes from, check out the original SAC paper (arXiv 1801.01290) and look
    in appendix C. This is a more numerically-stable equivalent to Eq 21.
    Try deriving it yourself as a (very difficult) exercise. :)
    """
    logp_pi -= tf.reduce_sum(
        2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


def mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation=activation)
        for size in hidden_sizes
    ], name)


def layer_norm_mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_sizes[0]),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation(tf.nn.tanh),
        mlp(hidden_sizes[1:], activation)
    ], name)


class MLPActorCriticFactory:
    """Factory of MLP stochastic actors and critics.

    Args:
        observation_space (gym.spaces.Box): A continuous observation space
          specification.
        action_space (gym.spaces.Box): A continuous action space
          specification.
        hidden_sizes (list): A hidden layers shape specification.
        activation (tf.function): A hidden layers activations specification.
        ac_number (int): Number of the actor-critic models in the ensemble.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes,
        activation,
        ac_number,
    ):
        self._obs_dim = observation_space.shape[0]
        self._act_dim = action_space.shape[0]
        self._act_scale = action_space.high[0]
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._ac_number = ac_number

    def _make_actor(self):
        """Constructs and returns the actor model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        body = mlp(self._hidden_sizes, self._activation)(obs_input)
        mu = tf.keras.layers.Dense(self._act_dim)(body)
        log_std = tf.keras.layers.Dense(self._act_dim)(body)

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(input=mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        # Put the actions in the limit.
        mu = mu * self._act_scale
        pi = pi * self._act_scale

        return tf.keras.Model(inputs=obs_input, outputs=[mu, pi, logp_pi])

    def make_actor(self):
        """Constructs and returns the ensemble of actor models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        mus, pis, logp_pis = [], [], []
        for obs_input in tf.unstack(obs_inputs, axis=0):
            model = self._make_actor()
            mu, pi, logp_pi = model(obs_input)
            mus.append(mu)
            pis.append(pi)
            logp_pis.append(logp_pi)
        return tf.keras.Model(inputs=obs_inputs, outputs=[
            tf.stack(mus, axis=0),
            tf.stack(pis, axis=0),
            tf.stack(logp_pis, axis=0),
        ])

    def _make_critic(self):
        """Constructs and returns the critic model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        act_input = tf.keras.Input(shape=(self._act_dim,))

        concat_input = tf.keras.layers.Concatenate(
            axis=-1)([obs_input, act_input])

        q = tf.keras.Sequential([
            mlp(self._hidden_sizes, self._activation),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Reshape([])  # Very important to squeeze values!
        ])(concat_input)

        return tf.keras.Model(inputs=[obs_input, act_input], outputs=q)

    def make_critic(self):
        """Constructs and returns the ensemble of critic models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        act_inputs = tf.keras.Input(shape=(None, self._act_dim),
                                    batch_size=self._ac_number)
        qs = []
        for obs_input, act_input in zip(tf.unstack(obs_inputs, axis=0),
                                        tf.unstack(act_inputs, axis=0)):
            model = self._make_critic()
            q = model([obs_input, act_input])
            qs.append(q)
        return tf.keras.Model(inputs=[obs_inputs, act_inputs],
                              outputs=tf.stack(qs, axis=0))
