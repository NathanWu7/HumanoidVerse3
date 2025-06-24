from humanoidverse.envs.base_task.term import base
import numpy as np
import torch
from loguru import logger

class LimitsCurrculum(base.BaseManager):

    def __init__(self, _task):
        super(LimitsCurrculum, self).__init__(_task)

    # stage 1
    def init(self):

        if not hasattr(self.task, "rewards_manager"):
            return

        rewards_manager = self.task.rewards_manager
        if not hasattr(rewards_manager, 'actuators_rewards'):
            return

        if 'reward_limit' not in self.config.rewards or 'reward_limits_curriculum' not in self.config.rewards.reward_limit:
            self.use_reward_limits_dof_pos_curriculum = False
            self.use_reward_limits_dof_vel_curriculum = False
            self.use_reward_limits_torque_curriculum = False
            return

        self.use_reward_limits_dof_pos_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum
        self.use_reward_limits_dof_vel_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum
        self.use_reward_limits_torque_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum

        if self.use_reward_limits_dof_pos_curriculum:
            logger.info(f"Use Reward Limits DOF Curriculum: {self.use_reward_limits_dof_pos_curriculum}")
            logger.info(f"Reward Limits DOF Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit}")
            logger.info(f"Reward Limits DOF Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit}")
            logger.info(f"Reward Limits DOF Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit}")
            self.soft_dof_pos_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit

        if self.use_reward_limits_dof_vel_curriculum:
            logger.info(f"Use Reward Limits DOF Vel Curriculum: {self.use_reward_limits_dof_vel_curriculum}")
            logger.info(f"Reward Limits DOF Vel Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit}")
            self.soft_dof_vel_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit

        if self.use_reward_limits_torque_curriculum:
            logger.info(f"Use Reward Limits Torque Curriculum: {self.use_reward_limits_torque_curriculum}")
            logger.info(f"Reward Limits Torque Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit}")
            logger.info(f"Reward Limits Torque Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit}")
            logger.info(f"Reward Limits Torque Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit}")
            self.soft_torque_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit

    def reset(self, env_ids):
        if 0 == len(env_ids):
            return

        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager

        if not hasattr(self.task, "rewards_manager"):
            return

        rewards_manager = self.task.rewards_manager
        if not hasattr(rewards_manager, 'actuators_rewards'):
            return

        actuators_rewards = rewards_manager.actuators_rewards

        if self.use_reward_limits_dof_pos_curriculum:
            if episode_manager.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_down_threshold:
                self.soft_dof_pos_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            elif episode_manager.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_up_threshold:
                self.soft_dof_pos_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            self.soft_dof_pos_curriculum_value = np.clip(self.soft_dof_pos_curriculum_value,
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit,
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit)
            ## update
            m = (self.task.simulator.hard_dof_pos_limits[:, 0: 1] + self.task.simulator.hard_dof_pos_limits[:, 1:]) / 2
            r = self.task.simulator.hard_dof_pos_limits[:, 1: ] - self.task.simulator.hard_dof_pos_limits[:, 0: 1]
            lower_soft_limit = m - 0.5 * r * self.soft_dof_pos_curriculum_value
            upper_soft_limit = m + 0.5 * r * self.soft_dof_pos_curriculum_value

            actuators_rewards.limits_dof_pos = torch.cat((lower_soft_limit, upper_soft_limit), dim = -1)

        if self.use_reward_limits_dof_vel_curriculum:
            if episode_manager.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_down_threshold:
                self.soft_dof_vel_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            elif episode_manager.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_up_threshold:
                self.soft_dof_vel_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            self.soft_dof_vel_curriculum_value = np.clip(self.soft_dof_vel_curriculum_value,
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit,
                                                         self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit)
            ## update
            actuators_rewards.limits_dof_vel = self.task.simulator.dof_vel_limits * self.soft_dof_vel_curriculum_value


        if self.use_reward_limits_torque_curriculum:
            if episode_manager.average_episode_length < self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_down_threshold:
                self.soft_torque_curriculum_value *= (1 + self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            elif episode_manager.average_episode_length > self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_level_up_threshold:
                self.soft_torque_curriculum_value *= (1 - self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            self.soft_torque_curriculum_value = np.clip(self.soft_torque_curriculum_value,
                                                        self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit,
                                                        self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit)
            ## update
            actuators_rewards.limits_torque = self.task.simulator.torque_limits * self.soft_torque_curriculum_value

    def post_compute(self):
        if not hasattr(self.task, "extras_manager"):
            return

        if not hasattr(self.task, "rewards_manager"):
            return

        rewards_manager = self.task.rewards_manager
        if not hasattr(rewards_manager, 'actuators_rewards'):
            return

        extras_manager = self.task.extras_manager
        if self.use_reward_limits_dof_pos_curriculum:
            extras_manager.log_dict["soft_dof_pos_curriculum_value"] = torch.tensor(self.soft_dof_pos_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_dof_vel_curriculum:
            extras_manager.log_dict["soft_dof_vel_curriculum_value"] = torch.tensor(self.soft_dof_vel_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_torque_curriculum:
            extras_manager.log_dict["soft_torque_curriculum_value"] = torch.tensor(self.soft_torque_curriculum_value, dtype=torch.float)
