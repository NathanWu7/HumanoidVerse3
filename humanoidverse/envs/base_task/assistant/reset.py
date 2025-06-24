import torch
from humanoidverse.envs.base_task.term import base
import numpy as np
from loguru import logger

'''
NOTE it not regist to register,
is episode manager's member.
'''


class Reset(base.BaseManager):
    def __init__(self, _task):
        super(Reset, self).__init__(_task)
        self.reset_status = {}

    # stage 1
    def init(self):
        super(Reset, self).pre_init()
        self._collect_compute_reset()

    def pre_compute(self):
        self.reset_status.clear()


    def compute_reset(self):
        episode_manager = self.task.episode_manager
        # termination
        for _key, _termination in self.check_termination_dict.items():
            _reset = _termination()
            if _reset is None:
                continue

            self.reset_status[_key] = torch.sum(_reset.float())
            episode_manager.termination_buf |= _reset

        # time_out
        for _key, _time_out in self.check_time_out_dict.items():
            _reset = _time_out()
            if _reset is None:
                continue
            self.reset_status[_key] = torch.sum(_reset.float())
            episode_manager.time_out_buf |= _reset

    def post_compute(self):
        assert hasattr(self.task, "extras_manager")
        extras_manager = self.task.extras_manager

        for _key, _value in self.reset_status.items():
            extras_manager.log_dict[_key] = _value

    def _collect_compute_reset(self):
        ## callect termination
        _termination_dicts = {}
        _time_out_dicts = {}
        for _manager in self.task.managers.values():
            _items = dir(_manager)
            for _item in _items:
                if _item.startswith("_check_terminate"):
                    _termination_dicts[_item] = getattr(_manager, _item)
                    logger.info(f"valid reset(termination): {_item[7:]}")
                elif _item.startswith("_check_time_out"):
                    _time_out_dicts[_item] = getattr(_manager, _item)
                    logger.info(f"valid reset(time_out): {_item[7:]}")



        ## check termination
        self.check_termination_dict = {}
        self.check_time_out_dict = {}

        import copy
        terminations = copy.deepcopy(self.config.termination)
        # remove zero scales + multiply non-zero ones by dt
        for key in list(terminations.keys()):
            _valid = terminations[key]
            if not _valid:
                logger.info(f"unuse reset(termination/time_out): {key} ")
                continue

            _func_key = f'_check_{key}'
            if key.startswith('terminate'):
                if _func_key in _termination_dicts:
                    self.check_termination_dict[key] = _termination_dicts[_func_key]
                    logger.info(f"using reset(termination): {key}")
                else:
                    logger.warning(f"reset(termination): {key} no implement")
            elif key.startswith('time_out'):
                if _func_key in _time_out_dicts:
                    self.check_time_out_dict[key] = _time_out_dicts[_func_key]
                    logger.info(f"using reset(time_out): {key}")
                else:
                    logger.warning(f"reset(time_out): {key} no implement")
