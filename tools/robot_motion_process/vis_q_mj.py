import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())


# from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif keycode == 256 or chr(keycode) == "Q":
        print("Esc")
        os._exit(0)
    elif chr(keycode) == 'L':
        speed = speed * 1.5
        print("Speed: ", speed)
    elif chr(keycode) == 'K':
        speed = speed / 1.5
        print("Speed: ", speed)
    elif chr(keycode) == 'J':
        print("Toggle Rewind: ", not rewind)
        rewind = not rewind
    elif keycode == 262: #(Right)
        time_step+=dt
    elif keycode == 263: #(Left)
        time_step-=dt
    elif chr(keycode) == "Q":
        print('Modify left foot contact!!!')
        contact_mask[curr_time][0] = 1. - contact_mask[curr_time][0]
        resave = True
    elif chr(keycode) == "E":
        print('Modify right foot contact!!!')
        contact_mask[curr_time][1] = 1. - contact_mask[curr_time][1]
        resave = True
    else:
        print("not mapped", chr(keycode), keycode)
    
    
        
@hydra.main(version_base=None)
def main(cfg : DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind, motion_data_keys, contact_mask, curr_time, resave
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind \
        = 0, 1, 0, set(), 0, 1/30, 1.0, False, False
    # if 'dt' in cfg:
    #     dt = cfg.dt
    motion_file = cfg.motion_file
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    curr_motion_key = motion_data_keys[motion_id]
    curr_motion = motion_data[curr_motion_key]
    print(motion_file)
    
    speed = 1.0 if 'speed' not in cfg else cfg.speed
    hang = False if 'hang' not in cfg else cfg.hang
    if 'fps' in curr_motion:
        dt = 1.0 / curr_motion['fps']
    elif 'dt' in cfg:
        dt = cfg.dt
        
    print("Motion file: ", motion_file)
    print("Motion length: ", motion_data[motion_data_keys[0]]['dof'].shape[0], 'frames')
    print("Speed: ", speed)
    print()

    if 'contact_mask' in curr_motion.keys():
        contact_mask = curr_motion['contact_mask']
    else:
        contact_mask = None
    curr_time = 0
    resave = False


    humanoid_xml = "./description/robots/g1/g1_23dof_lock_wrist.xml"
    print(humanoid_xml)
    
    vis_smpl = False if 'vis_smpl' not in cfg else cfg.vis_smpl
    vis_tau_key = 'tau' if 'vis_tau_key' not in cfg else cfg.vis_tau_key
    vis_tau = vis_tau_key in curr_motion if 'vis_tau' not in cfg else cfg.vis_tau
    vis_contact = 'contact_mask' in curr_motion if 'vis_contact' not in cfg else cfg.vis_contact
    
    if vis_smpl: assert 'smpl_joints' in curr_motion
    if vis_tau: assert vis_tau_key in curr_motion and not vis_contact
    if vis_contact: assert 'contact_mask' in curr_motion and not vis_tau

    if not vis_smpl:
        cfg_robot = OmegaConf.load("description/robots/g1/phc_g1_23dof.yaml")
        humanoid_fk = Humanoid_Batch(cfg_robot)  # load forward kinematics model
        pose_aa = torch.from_numpy(curr_motion['pose_aa']).unsqueeze(0)
        root_trans = torch.from_numpy(curr_motion['root_trans_offset']).unsqueeze(0)
        fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)
        joint_gt = fk_return.global_translation_extend[0]
    
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt
    
    print("Init Pose: ",(np.array(np.concatenate(
        [curr_motion['root_trans_offset'][0],curr_motion['root_rot'][0][[3, 0, 1, 2]], curr_motion['dof'][0]]
        ),dtype=np.float32)).__repr__())
    
    # breakpoint()
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        
        viewer.cam.lookat[:] = np.array([0,0,0.7])
        viewer.cam.distance = 3.0        
        viewer.cam.azimuth = 180         
        viewer.cam.elevation = -30                      # 负值表示从上往下看viewer
        
        for _ in range(50):
                # not display the ball
            # add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 0]))
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        
        # breakpoint()
        while viewer.is_running():
            step_start = time.time()
            if time_step >= curr_motion['dof'].shape[0]*dt:
                time_step -= curr_motion['dof'].shape[0]*dt
            curr_time = round(time_step/dt) % curr_motion['dof'].shape[0]
            
            if hang:
                mj_data.qpos[:3] = np.array([0,0,0.8])
            else:
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]] #xyzw 2 wxyz
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]
            
            
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt * (1 if not rewind else -1) * speed
            
                
            if vis_smpl:
                joint_gt = motion_data[curr_motion_key]['smpl_joints']
                if not np.all(joint_gt[curr_time] == 0):
                    for i in range(joint_gt.shape[1]):
                        viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
            else:
                for i in range(23):
                    viewer.user_scn.geoms[i+1].pos = joint_gt[curr_time, i+1]
            
            if vis_contact: 
                viewer.user_scn.geoms[6].rgba = np.array([0, 1-curr_motion['contact_mask'][curr_time, 0], 0, 1])
                viewer.user_scn.geoms[12].rgba = np.array([0, 1-curr_motion['contact_mask'][curr_time, 1], 0, 1])
                
            if vis_tau:
                scale_factor = 0.1
                for i in range(23):
                    tau = curr_motion[vis_tau_key][curr_time, i]
                    color_gradient = abs(tau) * scale_factor
                    if tau > 0:
                        viewer.user_scn.geoms[i+1].rgba = np.array([0.8,0.1,0.1,0.1+color_gradient])
                        # viewer.user_scn.geoms[i+1].rgba = np.array([0.1+color_gradient,0.,0.,1.])
                    elif tau < 0:
                        viewer.user_scn.geoms[i+1].rgba = np.array([0.1,0.8,0.1,0.1+color_gradient])
                        # viewer.user_scn.geoms[i+1].rgba = np.array([0,0.1+color_gradient,0.,1.])
                        
            
                

            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
            print("Frame ID: ",curr_time,'\t | Times ',f"{time_step:4f}",end='\r\b')

    if resave:
        motion_data[curr_motion_key]['contact_mask'] = contact_mask
        motion_file = motion_file.split('.')[0]+'_edit_cont.pkl'
        print(motion_file)
        joblib.dump(motion_data, motion_file)

if __name__ == "__main__":
    main()
