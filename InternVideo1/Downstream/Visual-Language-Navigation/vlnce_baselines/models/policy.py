import abc
from typing import Any

from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import (
    CategoricalNet,
    CustomFixedCategorical,
)
from torch.distributions import Categorical


class ILPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        r"""Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions

        # self.action_distribution = CategoricalNet(
        #     self.net.output_size, self.dim_actions
        # )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        print('need to revise for CMA and VLNBERT')
        import pdb; pdb.set_trace()

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        # if distribution.logit
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_hidden_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.action_distribution(features)

    def act2(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        print('need to revise for CMA and VLNBERT')
        import pdb; pdb.set_trace()

        feature_rgb, feature_depth, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution_rgb = self.action_distribution(feature_rgb)
        distribution_depth = self.action_distribution(feature_depth)

        probs = (distribution_rgb.probs + distribution_depth.probs)/2
        # if distribution.logit
        if deterministic:
            action = probs.argmax(dim=-1, keepdim=True)
        else:
            action = Categorical(probs).sample().unsqueeze(-1)

        return action, rnn_hidden_states
