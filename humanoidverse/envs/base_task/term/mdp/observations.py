from humanoidverse.envs.base_task.term import base
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.envs.base_task.assistant import history
import copy
import torch

class BaseObservations(base.BaseManager):
    def __init__(self, _task):
        super(BaseObservations, self).__init__(_task)
        self.dim_obs = self.config.robot.policy_obs_dim
        self.dim_critic_obs = self.config.robot.critic_obs_dim
        self.noise_scales = copy.deepcopy(self.config.obs.noise_scales)


    def init(self):
        self.obs_buf_dict = {}
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}
        self.history_manager = history.HistoryManager(self.task)

    def post_init(self):
        super(BaseObservations, self).init()
        self._collect_observationss()

    # stage 3
    def pre_step(self):
        self.hist_obs_dict.clear()

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.history_manager.reset(env_ids)

    def compute(self):
        """ Computes observations
        """
        # compute Algo observations
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            self.obs_buf_dict_raw[obs_key] = dict()
            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.config.obs.obs_scales, self.noise_scales)

        if True:
            history_manager = self.history_manager
            # Compute history observations
            history_obs_list = history_manager.history_handler.history.keys()
            parse_observation(self, history_obs_list, self.hist_obs_dict, self.config.obs.obs_scales, self.noise_scales)

        self._post_config_observation_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

    def post_compute(self):
        self.history_manager.post_compute()

    # helper functions
    def _post_config_observation_callback(self):
        self.obs_buf_dict = dict()
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            obs_keys = sorted(obs_config)
            # print("obs_keys", obs_keys)
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)


    def _collect_observationss(self):
        self.observationss_dict = {}
        for _key, _manager in self.task.managers.items():
            _items = dir(_manager)
            for _item in _items:
                if _item.startswith("_get_obs_"):
                    _obs_function = getattr(_manager, _item)
                    setattr(self, _item, _obs_function)

        _manager = self.history_manager
        _items = dir(_manager)
        for _item in _items:
            if _item.startswith("_get_obs_"):
                _obs_function = getattr(_manager, _item)
                setattr(self, _item, _obs_function)
