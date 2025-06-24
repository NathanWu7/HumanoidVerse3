
import torch
from humanoidverse.envs.base_task.term import base
from loguru import logger


class TrackTermination(base.BaseManager):
    def __init__(self, _task):
        super(TrackTermination, self).__init__(_task)

    def init(self):
        self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
        logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

    ## termination with motion_far
    def _check_terminate_when_motion_far(self):
        robotdata_manager = self.task.robotdata_manager
        robotstatus_manager = self.task.robotstatus_manager

        if not hasattr(robotdata_manager, 'current_motion_ref'):
            return None

        ref_body_pos = robotdata_manager.current_motion_ref["rg_pos_t"]
        ## diff compute - kinematic position
        _dif_global_body_pos = ref_body_pos - robotstatus_manager._rigid_body_pos

        reset_buf_motion_far = torch.any(torch.norm(_dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
        # TODO
        # self.reset_buf_motion_far = torch.sum(reset_buf_motion_far.float()).item()
        return reset_buf_motion_far
        # log current motion far threshold

    ## termination with timeout
    def _check_time_out_when_motion_end(self):
        assert hasattr(self.task, "episode_manager")
        episode_manager = self.task.episode_manager
        robotdata_manager = self.task.robotdata_manager
        current_time = (episode_manager.episode_length_buf) * self.task.dt + robotdata_manager.motion_start_times
        return current_time > robotdata_manager.motion_len
