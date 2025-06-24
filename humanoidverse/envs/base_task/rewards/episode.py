from humanoidverse.envs.base_task.term import base

class EpisodeRewards(base.BaseManager):
    def __init__(self, _task):
        super(EpisodeRewards, self).__init__(_task)

    ########################### PENALTY REWARDS ###########################
    def _reward_termination(self):
        _episode_manager = self.task.episode_manager
        # Terminal reward / penalty
        return _episode_manager.termination_buf
