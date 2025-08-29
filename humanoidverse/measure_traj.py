import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import omegaconf

# add argparse arguments
import toolz

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
import joblib
from pprint import pprint
from isaac_utils.rotations import *

Root_Path = Path(__file__).absolute().parent.parent

import json
import torch

global motion_lib 
motion_lib = None

def get_appendix_motion_data(pol_motion_file_path):
    with open(pol_motion_file_path, 'rb') as f:
        motion_data = joblib.load(f)
        assert len(motion_data) == 1, 'current only support single motion tracking'
        # get the first motion data
        motion_data = motion_data[next(iter(motion_data))]
        
    # print(motion_data.keys())
    # dict_keys(['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'actor_obs', 'action', 'terminate', 'root_lin_vel', 'root_ang_vel', 'dof_vel', 'motion_times', 'fps'])
    return {
        'action': torch.tensor(motion_data['action']),
        'actor_obs': torch.tensor(motion_data['actor_obs']),
        'terminate': torch.tensor(motion_data['terminate']),
        'motion_times': torch.tensor(motion_data['motion_times']),
        'fps': motion_data['fps'],
    }

    ...
    
def get_motionlib_data(motion_file_path, robot_cfg_path):
    robot_cfg = omegaconf.OmegaConf.load(robot_cfg_path)
    robot_cfg.robot.motion.motion_file = motion_file_path
    robot_cfg.robot.motion.asset.assetFileName = "g1_23dof_lock_wrist_fitmotionONLY.xml"
    robot_cfg.robot.motion.asset.assetRoot = Root_Path / "description/robots/g1/"
    
    from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX
    from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
    global motion_lib
    if motion_lib is None:
        # motion_lib = MotionLibRobot(robot_cfg.robot.motion, num_envs=1, device='cpu')
        motion_lib = MotionLibRobotWJX(robot_cfg.robot.motion, num_envs=1, device='cpu')
    else:
        motion_lib.load_data(motion_file_path)
    motion_data = motion_lib.load_motions(random_sample=False)[0]
    
    # print(motion_data.keys())
    # dict_keys(['global_velocity_extend', 'global_angular_velocity_extend', 'global_translation_extend', 'global_rotation_mat_extend', 'global_rotation_extend', 'global_translation', 'global_rotation_mat', 'global_rotation', 'local_rotation', 'global_root_velocity', 'global_root_angular_velocity', 'global_angular_velocity', 'global_velocity', 'dof_pos', 'dof_vels', 'fps', 'action'])
    return motion_data
    
def blend_motion(preblend_data, input_motion_times):
    def _calc_frame_blend(time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        
        
        frame_idx1 = torch.min(frame_idx0 + 1, torch.tensor(num_frames) - 1)
        
        
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend
    
    
    preblend_fps = preblend_data['fps']
    preblend_num_frames = preblend_data['dof_pos'].shape[0]
    preblend_motion_length = preblend_num_frames / preblend_fps
    # print(preblend_fps, preblend_motion_length, preblend_num_frames)
    
    # print(input_motion_times)
    if not torch.all(input_motion_times < preblend_motion_length):
        print(f"WARNING: input motion times should be less than preblend motion length: {input_motion_times.max()}, {preblend_motion_length}")
    
    frame_idx0, frame_idx1, blend = _calc_frame_blend(input_motion_times, preblend_motion_length, preblend_num_frames, 1/preblend_fps)
    # print(input_motion_times,frame_idx0, frame_idx1, blend)
    blend = blend.unsqueeze(-1)
    blended_data = {}
    
    for key in preblend_data.keys():
        if key == 'fps': 
            blended_data[key] = preblend_data[key]
            continue
        key_frame0 = preblend_data[key][frame_idx0]
        key_frame1 = preblend_data[key][frame_idx1]
        blend_exp = blend.reshape(-1, *([1] * (key_frame0.ndim - 1)))
        print(key, key_frame0.shape, key_frame1.shape, blend_exp.shape)
        if 'rotation' in key:
            if "mat" in key:
                quat_frame0 = matrix_to_quaternion(key_frame0)
                quat_frame1 = matrix_to_quaternion(key_frame1)
                blended = slerp(quat_frame0, quat_frame1, blend_exp)
                blended_data[key] = quaternion_to_matrix(blended)
            else:
                # quaternion, slerp
                blended = slerp(key_frame0, key_frame1, blend_exp)
                ...
            blended_data[key] = blended
            # print(key, 'SLERP', preblend_data[key].shape)
            # frame0 = sRot.from_matrix(preblend_data[key][frame_idx0])
        else:
            blended = blend_exp * key_frame1 + (1 - blend_exp) * key_frame0
            blended_data[key] = blended
            # print(key)
    return blended_data
    ...
    
def load_traj_data(pol_motion_file_path, ref_motion_file_path, robot_cfg_path):
    pol_appendix = get_appendix_motion_data(pol_motion_file_path)
    pol_motion_data = get_motionlib_data(pol_motion_file_path, robot_cfg_path)
    ref_motion_data_preblend = get_motionlib_data(ref_motion_file_path, robot_cfg_path)
    
    print("Motion Length: ", pol_appendix['motion_times'].shape[0] / pol_appendix['fps'])
    print("Motion Num Frames: ", pol_appendix['motion_times'].shape[0])
    
    # the ref motion data is 30fps typically, but the pol motion data is 50fps (or more)
    # so we need to blend the motion data to the same fps
    ref_motion_data = blend_motion(ref_motion_data_preblend, pol_appendix['motion_times'])

    return {
        'pol': pol_motion_data,
        'ref': ref_motion_data,
        'appendix': pol_appendix,
    }

def eval_accuracy(traj_data,delta_per_frame=False):
    """
    We evaluate policy’s ability to imitate the reference motion
by comparing the tracking error of the global body position
Eg-mpjpe (mm), the root-relative mean per-joint (MPJPE) Empjpe mm), acceleration error Eacc (mm/frame2), and root velocity
Evel (mm/frame). 
    """
    pol = traj_data['pol']
    ref = traj_data['ref']
    appendix = traj_data['appendix']
    
    # compute the global mpjpe
    gmpbpe = torch.norm(pol['global_translation'] - ref['global_translation'], dim=-1).mean(dim=-1).mean()
    
    
    # compute the root-relative mpjpe
    root_relative_position = pol['global_translation'] - pol['global_translation'][..., 0:1, :]
    root_relative_position_ref = ref['global_translation'] - ref['global_translation'][..., 0:1, :]
    mpbpe = torch.norm(root_relative_position - root_relative_position_ref, dim=-1).mean(dim=-1).mean()
    
    # compute the dof mpjpe
    dof_mpjpe = torch.norm(pol['dof_pos'] - ref['dof_pos'], dim=-1).mean(dim=-1).mean()
    
    # compute the acceleration error
    if delta_per_frame:
        delta = 1
    else:
        delta = traj_data['appendix']['fps']
        
    pol_dof_vel = (pol['dof_pos'][1:] - pol['dof_pos'][:-1]) * delta
    ref_dof_vel = (ref['dof_pos'][1:] - ref['dof_pos'][:-1]) * delta
    
    pol_dof_acc = (pol_dof_vel[1:] - pol_dof_vel[:-1]) * delta
    ref_dof_acc = (ref_dof_vel[1:] - ref_dof_vel[:-1]) * delta
    
    dof_vel_error = torch.norm(pol_dof_vel - ref_dof_vel, dim=-1).mean(dim=-1).mean()
    dof_acc_error = torch.norm(pol_dof_acc - ref_dof_acc, dim=-1).mean(dim=-1).mean()
        
        
    pol_vel = (pol['global_translation'][1:] - pol['global_translation'][:-1]) * delta
    ref_vel = (ref['global_translation'][1:] - ref['global_translation'][:-1]) * delta
    
    pol_acc = (pol_vel[1:] - pol_vel[:-1]) * delta
    ref_acc = (ref_vel[1:] - ref_vel[:-1]) * delta
    
    acceleration_error = torch.norm(pol_acc - ref_acc, dim=-1).mean(dim=-1).mean()
    velocity_error = torch.norm(pol_vel - ref_vel, dim=-1).mean(dim=-1).mean()
    root_acceleration_error = torch.norm(pol_acc[..., 0:1, :] - ref_acc[..., 0:1, :], dim=-1).mean(dim=-1).mean()
    root_velocity_error = torch.norm(pol_vel[..., 0:1, :] - ref_vel[..., 0:1, :], dim=-1).mean(dim=-1).mean()
    
    if 'contact_mask' in pol and 'contact_mask' in ref:
        contact_acc = torch.mean((pol['contact_mask'] - ref['contact_mask']).abs(), dim=-1).mean()
    
    # # pol_acc = (pol['global_velocity'][1:] - pol['global_velocity'][:-1]) * traj_data['appendix']['fps']
    # ref_acc = (ref['global_velocity'][1:] - ref['global_velocity'][:-1]) * traj_data['appendix']['fps']
    # acceleration_error = torch.norm(pol_acc - ref_acc, dim=-1).mean(dim=-1).mean()
    # 
    # # compute the root velocity
    # root_velocity = pol['global_velocity'][..., 0:1, :]
    # root_velocity_ref = ref['global_velocity'][..., 0:1, :]
    
    # breakpoint()
    
    resdict = {
        'E_gmpbpe': gmpbpe,
        'E_mpbpe': mpbpe,
        'E_mpjpe': dof_mpjpe,
        'E_mpjve': dof_vel_error,
        'E_mpjae': dof_acc_error,
        'E_pbve': velocity_error,
        'E_pbae': acceleration_error,
        'E_root_acc': root_acceleration_error,
        'E_root_vel': root_velocity_error,
        
    }
    if 'contact_mask' in pol and 'contact_mask' in ref:
        resdict['E_contact_acc'] = contact_acc
    return resdict
    
def eval_smoothness(traj_data, delta_per_frame=False):
    pol = traj_data['pol']
    ref = traj_data['ref']
    appendix = traj_data['appendix']
    
    if delta_per_frame:
        delta = 1
    else:
        delta = traj_data['appendix']['fps']
        
    diff_fn = lambda x: (x[1:] - x[:-1]) * delta
        
    # compute the smoothness
    pol_pos = pol['global_translation']
    pol_vel = diff_fn(pol_pos)
    pol_acc = diff_fn(pol_vel)
    pol_jerk = diff_fn(pol_acc)
    
    L2_vel = torch.norm(pol_vel, dim=-1).mean(dim=-1).mean()
    L2_acc = torch.norm(pol_acc, dim=-1).mean(dim=-1).mean()
    L2_jerk = torch.norm(pol_jerk, dim=-1).mean(dim=-1).mean()
    
    pol_dof_pos = pol['dof_pos']
    pol_dof_vel = diff_fn(pol_dof_pos)
    pol_dof_acc = diff_fn(pol_dof_vel)
    pol_dof_jerk = diff_fn(pol_dof_acc)
    
    L2_dof_vel = torch.norm(pol_dof_vel, dim=-1).mean(dim=-1).mean()
    L2_dof_acc = torch.norm(pol_dof_acc, dim=-1).mean(dim=-1).mean()
    L2_dof_jerk = torch.norm(pol_dof_jerk, dim=-1).mean(dim=-1).mean()
    
    ref_pos = ref['global_translation']
    ref_vel = diff_fn(ref_pos)
    ref_acc = diff_fn(ref_vel)
    ref_jerk = diff_fn(ref_acc)
    
    L2_ref_vel = torch.norm(ref_vel, dim=-1).mean(dim=-1).mean()
    L2_ref_acc = torch.norm(ref_acc, dim=-1).mean(dim=-1).mean()
    L2_ref_jerk = torch.norm(ref_jerk, dim=-1).mean(dim=-1).mean()
    
    ref_dof_pos = ref['dof_pos']
    ref_dof_vel = diff_fn(ref_dof_pos)
    ref_dof_acc = diff_fn(ref_dof_vel)
    ref_dof_jerk = diff_fn(ref_dof_acc)
    
    L2_ref_dof_vel = torch.norm(ref_dof_vel, dim=-1).mean(dim=-1).mean()
    L2_ref_dof_acc = torch.norm(ref_dof_acc, dim=-1).mean(dim=-1).mean()
    L2_ref_dof_jerk = torch.norm(ref_dof_jerk, dim=-1).mean(dim=-1).mean()
    
    # smoothness = torch.norm(pol['global_velocity'] - ref['global_velocity'], dim=-1).mean(dim=-1).mean()
    return {
        'L2_vel': L2_vel,
        'L2_acc': L2_acc,
        'L2_jerk': L2_jerk,
        'L2_dof_vel': L2_dof_vel,
        'L2_dof_acc': L2_dof_acc,
        'L2_dof_jerk': L2_dof_jerk,
        'L2_ref_vel': L2_ref_vel,
        'L2_ref_acc': L2_ref_acc,
        'L2_ref_jerk': L2_ref_jerk,
        'L2_ref_dof_vel': L2_ref_dof_vel,
        'L2_ref_dof_acc': L2_ref_dof_acc,
        'L2_ref_dof_jerk': L2_ref_dof_jerk,
    }
    ...
    
apply_1e3 = lambda x: {k: v * 1e3 for k, v in x.items()}
