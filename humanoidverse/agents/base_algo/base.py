from agents.base_algo import base_algo
from enum import Enum

class Context(Enum):
    Error = 0
    ROLLOUT = 1
    TRAIN = 2
    EVAL = 3

class BaseComponent():
    def __init__(self, _algo: base_algo.BaseAlgo):
        self.algo = _algo
        self.num_envs = _algo.env.num_envs
        self.device = _algo.device
        self.config = _algo.config

    # stage 1
    def pre_init(self):
        pass

    def init(self):
        pass

    def post_init(self):
        pass

    # stage 2
    # level 0
    def pre_loop(self, _inputs = None):
        pass

    def post_loop(self, _inputs = None):
        pass

    # level 1
    def pre_epoch(self, _inputs = None):
        pass

    def post_epoch(self, _inputs = None):
        pass

    # level 2
    def pre_step(self, _inputs = None):
        pass

    def step(self, _inputs = None):
        pass

    def post_step(self, _inputs = None):
        pass

    # help function
    def load(self, _loaded_dict: dict):
        pass

    '''
    for save state
    '''
    def update(self, _state_dict: dict):
        pass

    '''
    for write state
    '''
    def write(self, writer, _it):
        pass

    '''
    for write state
    '''
    def write(self, writer, _it):
        pass

    '''
    only for omponent which need to output log
    def log_epoch(self, width, pad):
        pass

    '''
