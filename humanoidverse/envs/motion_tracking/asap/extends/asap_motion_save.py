from humanoidverse.envs.base_task.term import base
from loguru import logger
import numpy as np
import torch

class MotionSave(base.BaseManager):

    def __init__(self, _task):
        super(MotionSave, self).__init__(_task)
