import os

from gym_unity.envs import ActionFlattener
import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import QNetwork


class TeamAgent(AgentInterface):

    def __init__(self, env):
        # use flattened, Discrete actions instead of default MultiDiscrete
        self.flattener = ActionFlattener(env.action_space.nvec)
        # this agent's model works with team_vs_policy variation of the env
        # so we need to convert observations & actions
        self.model = QNetwork(
            env.observation_space.shape[0],
            self.flattener.action_space.n,
            seed=0,
        )
        # check if weights exist, load weights & put model in eval mode
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),  "checkpoint.pth"
        )

        if os.path.isfile(weights_path):
            self.model.load_state_dict(torch.load(weights_path))
        else:
            print("Checkpoint not found.")
        self.model.eval()
        self.name = "TeamAgent"

    def act(self, observation):
        actions = {}
        # for each team player
        for player_id in observation:
            # create state tensor & feed it to model
            state = torch.from_numpy(observation[player_id]).float().unsqueeze(0)
            action_values = self.model(state)
            action = np.argmax(action_values.data.numpy())
            # convert Discrete action index to MultiDiscrete
            actions[player_id] = self.flattener.lookup_action(action)
        return actions