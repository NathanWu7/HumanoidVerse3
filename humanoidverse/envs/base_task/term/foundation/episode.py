import torch
from humanoidverse.envs.base_task.term import base
from humanoidverse.envs.base_task.assistant import reset
import numpy as np

class BaseEpisode(base.BaseManager):
    def __init__(self, _task):
        super(BaseEpisode, self).__init__(_task)
        self.max_episode_length_s = self.config.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.task.dt)
        self.reset_manager = reset.Reset(self.task)

    # stage 1
    def init(self):
        # time out flags
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.task_data.time_out_buf = self.time_out_buf
        # termination flags
        self.termination_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.task_data.termination_buf = self.termination_buf
        # episode length
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.common_step_counter = torch.tensor(0, device=self.device, dtype=torch.long)

        # for reward penalty curriculum
        # NOTE it is used after reset
        self.num_compute_average_epl = self.config.rewards.num_compute_average_epl
        self.average_episode_length = torch.tensor(0, device=self.device, dtype=torch.long) # num_compute_average_epl last termination episode length
        self.reset_manager.init()

    ## stage 2
    def pre_step(self):
        self.time_out_buf[:] = 0
        self.termination_buf[:] = 0

        self.common_step_counter  +=1
        # it will reset to 0
        self.episode_length_buf += 1
        # keep status for later calcute
        self.reset_manager.pre_step()

    def compute_reset(self):
        super(BaseEpisode, self).compute_reset()
        """ Check if environments need to be reset
        """
        # termination
        self.reset_manager.compute_reset()
        self._time_out()

        env_ids = self.reset_env_ids
        if len(env_ids) == 0:
            return

        # calcute average_episode_length first
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)
        # reset average_episode_length later
        self.episode_length_buf[env_ids] = 0

    ## help function
    @property
    def reset_buf(self):
        '''
        called only for compute & post_compute & post_step
        '''
        return self.time_out_buf | self.termination_buf

    @property
    def reset_env_ids(self):
        return self.reset_buf.nonzero(as_tuple=False).flatten()

    def post_compute(self):
        self.reset_manager.post_compute()

    ## called by agents
    def rand_episode_length(self):
        self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

    ###############################################################################
    def _time_out(self):
        # this functions is not manage by reset
        return self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
