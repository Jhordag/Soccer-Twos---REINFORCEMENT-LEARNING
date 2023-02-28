
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
import os

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_time_final",
        config={
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "ignore_worker_failures": True,
            "train_batch_size": 2000,
            "sgd_minibatch_size": 256,
            "lr": 1e-4,
            "lambda": .98,
            "gamma": .99,
            "clip_param": 0.2,
            "num_sgd_iter": 20,
            "rollout_fragment_length": 200,
            "model": {
                "fcnet_hiddens": [256,256],
                "vf_share_layers": False
            },
            "multiagent": {
                "policies": {
                    "attacker": (None, obs_space, act_space, {}),
                    "defender": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda agent_id: "attacker" if agent_id % 2 == 0 else "defender",
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
            },
        },
        stop={
            "timesteps_total": 15000000,  # 15M
            "time_total_s": 18000, # 4h
        },
        checkpoint_freq=20,
        checkpoint_at_end=True,
        local_dir="./results",
        #restore="./results/PPO_deepmind/PPO_Soccer_c2b61_00000_0_2023-02-27_11-12-25/checkpoint_000340/checkpoint-340",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
