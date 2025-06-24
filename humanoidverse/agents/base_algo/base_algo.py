import torch

from torch import nn, Tensor

import time
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict


from hydra.utils import instantiate

from humanoidverse.envs.base_task.base_task import BaseTask

class BaseAlgo:
    def __init__(self, env: BaseTask, config, device):
        self.env = env
        self.config = config
        self.device = device
        self._setup_component()

    def setup(self):
        return NotImplementedError

    def learn(self):
        return NotImplementedError

    def load(self, path):
        return NotImplementedError

    @property
    def inference_model(self):
        return NotImplementedError

    def env_step(self, actions, extra_info=None):
        obs_dict, rewards, dones, extras = self.env.step(actions, extra_info)
        return obs_dict, rewards, dones, extras

    @torch.no_grad()
    def evaluate_policy(self):
        return NotImplementedError

    def save(self, path=None, name="last.ckpt"):
        raise NotImplementedError

    ## helper function
    def _setup_component(self):
        self.components = {}
        from humanoidverse.agents.base_algo import register
        _registry = register.registry[self.namespace]
        for _name in _registry:
            _component = _registry[_name](self)
            self.components[_name] = _component
            setattr(self, _name, _component)
