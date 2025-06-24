from motion_lib.motion_lib_base import MotionLibBase
from motion_lib.torch_humanoid_batch import Humanoid_Batch
class MotionLibRobot(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device=None):  
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        self.output_device = device
        return

    def get_motion_length(self, motion_ids=None):
        if motion_ids is not None:
            motion_ids = motion_ids.cpu()
        _out = super(MotionLibRobot, self).get_motion_length(motion_ids)
        return _out.to(self.output_device)


    def sample_time(self, motion_ids, truncate_time=None):
        motion_ids = motion_ids.cpu()
        if truncate_time is not None:
            truncate_time = truncate_time.cpu()

        _out = super(MotionLibRobot, self).sample_time(motion_ids, truncate_time)
        return _out.to(self.output_device)

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_ids = motion_ids.cpu()
        motion_times = motion_times.cpu()
        if offset is not None:
            offset = offset.cpu()

        _out = super(MotionLibRobot, self).get_motion_state( motion_ids, motion_times, offset)
        for _key, _value in _out.items():
            _out[_key] = _value.to(self.output_device)

        return _out
