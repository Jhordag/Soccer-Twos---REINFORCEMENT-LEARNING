import ray
from ray import tune
from soccer_twos import EnvType
import gym
import numpy as np

from utils1 import create_rllib_env


NUM_ENVS_PER_WORKER = 3


if __name__ == "__main__":
    ray.init()

    def reward_function(agent,opponent,rew):
        score_reward = 0
        possession_reward = 0
        
        if agent.team_scored:
            score_reward = 2 
        elif opponent.team_scored:
            score_reward = -2
            
        if agent.ball_owned:
            possession_reward = 1 
        elif opponent.ball_owned:
            possession_reward = -0.5 
        
        return score_reward + possession_reward
        
    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = gym.spaces.Dict({
        "agent_0": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,)),
        "agent_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
    })
    act_space = gym.spaces.Dict({
        "agent_0": gym.spaces.Discrete(9),
        "agent_1": gym.spaces.Discrete(9)
    })

    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_time3",
        config={
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "multiagent": {
                "policies": {
                    "agent_0": (None, obs_space["agent_0"], act_space["agent_0"], {}),
                    "agent_1": (None, obs_space["agent_1"], act_space["agent_1"], {}),
                },
                "policy_mapping_fn": tune.function(lambda agent_id: agent_id),
                "policies_to_train": ["agent_0", "agent_1"],
                "policy_ops": ["compute_actions", "compute_gradients", "apply_gradients"],
                "observation_fn": lambda obs: {k: obs[k][0] for k in obs.keys()},
                "replay_mode": "lockstep",
                "check_valid_actions": True,
                "normalize_actions": False,
                "reward_fn": tune.function(reward_function),
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
            },
            "model": {
                "fcnet_hiddens": [512, 256],
                "fcnet_activation": "relu",
            },
            "num_sgd_iter": 10,
            "lr": 5e-5,
            "rollout_fragment_length": 100,
            "train_batch_size": 1000,
            "batch_mode": "truncate_episodes",
            "grad_clip": 0.5,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
        },
        stop={
            "timesteps_total": 15000000,
            "time_total_s": 18000,
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
