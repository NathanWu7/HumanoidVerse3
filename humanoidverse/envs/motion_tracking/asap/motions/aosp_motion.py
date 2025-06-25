from humanoidverse.envs.base_task.term import base
import torch
from tools.motion_lib import motion_lib_robot
from loguru import logger
import numpy as np


class AsapMotion(base.BaseManager):
    def __init__(self, _task):
        super(AsapMotion, self).__init__(_task)

    def pre_init(self):
        super(AsapMotion, self).pre_init()

        ## status
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

        ## motion
        self.config.robot.motion.step_dt = self.task.dt
        self.motion_lib = motion_lib_robot.MotionLibRobot(self.config.robot["motion"], self.num_envs, self.device)
        self.motion_lib.load_motions(random_sample=False)
        self.motion_len[:] = self.motion_lib.get_motion_length(self.motion_ids)

        ## for resample
        #self.motion_start_idx = 0
        self.num_motions = self.motion_lib._num_unique_motions
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.task.dt)

    def pre_compute(self):
        ## resample falg when training
        if self.config.resample_motion_when_training and not self.task.is_evaluating:
            episode_manager = self.task.episode_manager
            if episode_manager.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {episode_manager.common_step_counter}")
                ## update reset flag with True
                self.motion_lib.load_motions(random_sample=True)
                self.motion_len[:] = self.motion_lib.get_motion_length(self.motion_ids)
                episode_manager.time_out_buf[:] = 1

    def reset(self, env_ids):
        super(AsapMotion, self).reset(env_ids)
        if len(env_ids) != 0:
            if self.task.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
                self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
            else:
                self.motion_start_times[env_ids] = self.motion_lib.sample_time(self.motion_ids[env_ids])

    @property
    def motion_ref(self):
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager
        terrain_manager = self.task.terrain_manager

        offset = terrain_manager.env_origins
        motion_times = (episode_manager.episode_length_buf) * self.task.dt + self.motion_start_times # next frames so +1
        motion_ref = self.motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        return motion_ref

    @property
    def next_motion_ref(self):
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager
        terrain_manager = self.task.terrain_manager

        offset = terrain_manager.env_origins
        motion_times = (episode_manager.episode_length_buf + 1) * self.task.dt + self.motion_start_times # next frames so +1
        motion_ref = self.motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        return motion_ref

    @property
    def next_motion_times(self):
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager
        return (episode_manager.episode_length_buf + 1) * self.task.dt + self.motion_start_times # next frames so +1


class AsapMotionEvaluater(AsapMotion):
    def __init__(self, _task):
        super(AsapMotionEvaluater, self).__init__(_task)

    def pre_init(self):
        super(AsapMotionEvaluater, self).pre_init()
        self.motion_start_idx = 0
        self.num_motions = self.motion_lib._num_unique_motions
        self.next_motions_flags = False
        import threading
        self.flags_lock = threading.Lock()

    def next_task(self):
        self.flags_lock.acquire()  # 获取锁
        self.next_motions_flags = True
        self.flags_lock.release()  # 释放锁

    def pre_compute(self):
        super(AsapMotionEvaluater, self).pre_compute()
        _next_motions_flags = False
        self.flags_lock.acquire()  # 获取锁
        _next_motions_flags = self.next_motions_flags
        self.next_motions_flags = False
        self.flags_lock.release()  # 释放锁

        if _next_motions_flags:
            self.motion_start_idx += self.num_envs
            if self.motion_start_idx >= self.num_motions:
                self.motion_start_idx = 0

            self.motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
            self.motion_len[:] = self.motion_lib.get_motion_length(self.motion_ids)
            episode_manager = self.task.episode_manager
            episode_manager.time_out_buf[:] = 1
