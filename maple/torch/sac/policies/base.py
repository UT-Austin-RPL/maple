import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import maple.torch.pytorch_util as ptu
from maple.policies.base import ExplorationPolicy
from maple.torch.core import torch_ify, elem_or_tuple_to_numpy
from maple.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from maple.torch.networks import Mlp, CNN
from maple.torch.networks.basic import MultiInputSequential
from maple.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)


class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs_np, return_dist=False):
        info = {}
        if return_dist:
            actions, dist = self.get_actions(obs_np[None], return_dist=return_dist)
            info['dist'] = dist
        else:
            actions = self.get_actions(obs_np[None], return_dist=return_dist)
        return actions[0, :], info

    def get_actions(self, obs_np, return_dist=False):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        if return_dist:
            return elem_or_tuple_to_numpy(actions), dist
        else:
            return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist


class PolicyFromDistributionGenerator(
    MultiInputSequential,
    TorchStochasticPolicy,
):
    """
    Usage:
    ```
    distribution_generator = FancyGenerativeModel()
    policy = PolicyFromBatchDistributionModule(distribution_generator)
    ```
    """
    pass


class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist = self._action_distribution_generator.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())
