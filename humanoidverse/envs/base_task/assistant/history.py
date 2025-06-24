from humanoidverse.envs.base_task.term import base
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
import torch

'''
NOTE it not regist to register,
is 0bservations manager's member.
'''

class HistoryManager(base.BaseManager):
    def __init__(self, _task):
        super(HistoryManager, self).__init__(_task)
        self.history_handler = HistoryHandler(self.num_envs,
                                              self.config.obs.obs_auxiliary,
                                              self.config.obs.obs_dims,
                                              self.device)

    # stage 3
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.history_handler.reset(env_ids)

    def post_compute(self):
        if not hasattr(self.task, "observations_manager"):
            return

        _observations_manager = self.task.observations_manager

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, _observations_manager.hist_obs_dict[key])

    ######################### Observations #########################
    def _get_obs_history(self):
        assert "history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history']
        return self._get_history(history_config)

    def _get_obs_short_history(self):
        assert "short_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['short_history']
        return self._get_history(history_config)

    def _get_obs_long_history(self):
        assert "long_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['long_history']
        return self._get_history(history_config)

    ## actor
    def _get_obs_history_actor(self):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        return self._get_history(history_config)

    def _get_obs_history_critic(self):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        return self._get_history(history_config)

    def _get_history(self, history_config):
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # _b, _ndims * history_length
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
