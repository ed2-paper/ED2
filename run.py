"""Script for running experiments.

Configurations:
    agent (string): An agent full import path e.g. '@spinup_bis.agents.tf2.SAC'.
    task (string): OpenAI Gym environment name to train in.
    **: Any kwargs appropriate for the agent function you provided but `env_fn`.

Note:
    To pass functions or classes in kwargs, specify their full import path
    prefixed with '@' (at) character.
"""
import argparse
import importlib
import os
from pathlib import Path

import gym
import sklearn.model_selection as skl_ms

# Register our custom environments
import envs  # pylint: disable=unused-import


def parse_pyobject_configs(config):
    """Substitutes PyObject config entries with imported objects."""
    parsed_config = dict(config)
    for key, value in config.items():
        try:
            if value[0] == '@':
                module_name, obj_name = value[1:].rsplit('.', 1)
                module = importlib.import_module(module_name)
                parsed_config[key] = getattr(module, obj_name)
        except (TypeError, KeyError):
            pass  # Value is not a string.
    return parsed_config


def make_env_fn(env_name, env_kwargs):
    """Make the environment factory function."""
    robotics_envs = [
        'FetchPickAndPlace',
        'FetchPush',
        'FetchReach',
        'FetchSlide',
        'HandManipulateBlock',
        'HandManipulateEgg',
        'HandManipulatePen',
        'HandReach',
    ]

    is_robotics = False
    if any(x in env_name for x in robotics_envs):
        is_robotics = True

    def env_fn():
        env = gym.make(env_name, **env_kwargs)
        if is_robotics:
            env = gym.wrappers.FlattenObservation(
                gym.wrappers.FilterObservation(
                    env, ['observation', 'desired_goal']
                ))
        return env

    env_fn.__name__ = env_name
    return env_fn


def run(config):
    """Run an agent based on the specification."""
    config = parse_pyobject_configs(config)

    # Pop parameters from the configuration.
    agent = config.pop('agent')
    task = config.pop('task')
    task_kwargs = config.pop('task_kwargs', {})
    config.pop('experiment_id')  # Dismiss experiment id.

    # Run the agent.
    agent(make_env_fn(task, task_kwargs), **config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    vars_ = dict()
    with open(args.config) as f:
        exec(f.read(), vars_)

    params_grid = skl_ms.ParameterGrid(vars_['params_grid'])
    for idx, params in enumerate(params_grid):
        run({**vars_['base_config'], **params, 'experiment_id': idx})


if __name__ == '__main__':
    main()
