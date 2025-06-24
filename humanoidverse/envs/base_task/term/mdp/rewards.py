import torch
from humanoidverse.envs.base_task.term import base
from loguru import logger
from termcolor import colored
import copy
import numpy as np

class BaseRewardsManager(base.BaseManager):
    def __init__(self, _task):
        super(BaseRewardsManager, self).__init__(_task)


    # stage 1
    def pre_init(self):
        super(BaseRewardsManager, self).pre_init()
        self._collect_rewards()
        for _name, _rewards in self.rewards_dict.items():
            _rewards.pre_init()


    def init(self):
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        super(BaseRewardsManager, self).init()
        for _name, _rewards in self.rewards_dict.items():
            _rewards.init()

    def post_init(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        logger.info(colored(f"{self.config.rewards.set_reward} set reward on {self.config.rewards.set_reward_date}", "green"))

        for _name, _rewards in self.rewards_dict.items():
            _rewards.post_init()

        ### step 1 collect functions
        _rewards_functions = {}
        for _key, _rewards in self.rewards_dict.items():
            _items = dir(_rewards)
            for _item in _items:
                if _item.startswith("_reward_"):
                    _rewards_functions[_item] = getattr(_rewards, _item)

        ### step 2
        self.reward_scales = copy.deepcopy(self.config.rewards.reward_scales)
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            logger.info(f"Scale: {key} = {self.reward_scales[key]}")
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.task.dt

        ### step 3
        self.use_reward_penalty_curriculum = self.config.rewards.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.config.rewards.reward_initial_penalty_scale

        logger.info(colored(f"Use Reward Penalty: {self.use_reward_penalty_curriculum}", "green"))
        if self.use_reward_penalty_curriculum:
            logger.info(f"Penalty Reward Names: {self.config.rewards.reward_penalty_reward_names}")
            logger.info(f"Penalty Reward Initial Scale: {self.config.rewards.reward_initial_penalty_scale}")

        ### step 4
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for key_name, scale in self.reward_scales.items():

            name = '_reward_' + key_name
            _function = None
            if name in _rewards_functions:
                _function = _rewards_functions[name]

            if None == _function:
                logger.warning(f"Reward invalid: {name}")
                continue

            if key_name=="termination":
                self.reward_termination = _function
                continue

            self.reward_names.append(key_name)
            self.reward_functions.append(_function)

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                            for name in self.reward_scales.keys()}

    # stage 3
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # it must here
        self._pre_curriculum_compute()

        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb; ipdb.set_trace()
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales and hasattr(self, "reward_termination"):
            rew = self.reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def reset(self, env_ids):
        # fill extras
        # if not hasattr(self.task, "extras_manager") or not hasattr(self.task, "episode_manager"):
        #     return
        #
        if len(env_ids) == 0:
            return

        assert hasattr(self.task, "extras_manager")
        assert hasattr(self.task, "episode_manager")
        extras_manager = self.task.extras_manager
        episode_manager = self.task.episode_manager
        _rew_episode = {}

        for key in self.episode_sums.keys():
            _rew_episode['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / episode_manager.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        extras_manager.extras["episode"] = _rew_episode

    def post_compute(self):
        if not self.use_reward_penalty_curriculum:
            return

        assert hasattr(self.task, "extras_manager")
        extras_manager = self.task.extras_manager
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager

        extras_manager.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
        extras_manager.log_dict["average_episode_length"] = episode_manager.average_episode_length

    ## help functions
    def _pre_curriculum_compute(self):
        # episode_manager.average_episode_length update at compute_reset
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if not self.use_reward_penalty_curriculum:
            return

        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager

        if episode_manager.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
        elif episode_manager.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.config.rewards.reward_min_penalty_scale,
                                            self.config.rewards.reward_max_penalty_scale)

    def _collect_rewards(self):
        self.rewards_dict = {}
        for _name, _rewards in self.task.rewards_map.items():
            _rewards = _rewards(self.task)
            self.rewards_dict[_name] = _rewards
            setattr(self, _name, _rewards)
