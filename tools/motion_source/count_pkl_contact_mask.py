import glob
import os
import sys
import pdb
import os.path as osp
import numpy as np

sys.path.append(os.getcwd())

from utils.torch_humanoid_batch import Humanoid_Batch
import torch
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

from scipy.spatial.transform import Rotation as sRot

def foot_detect(positions, thres=0.002):
    fid_r, fid_l = 12,6
    positions = positions.numpy()
    velfactor, heightfactor = np.array([thres]), np.array([0.12]) 
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:,fid_l,2]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(int) & (feet_l_h < heightfactor).astype(int)).astype(np.float32)
    feet_l = np.expand_dims(feet_l,axis=1)
    feet_l = np.concatenate([np.array([[1.]]),feet_l],axis=0)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:,fid_r,2]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(int) & (feet_r_h < heightfactor).astype(int)).astype(np.float32)
    feet_r = np.expand_dims(feet_r,axis=1)
    feet_r = np.concatenate([np.array([[1.]]),feet_r],axis=0)
    return feet_l, feet_r

def count_pose_aa(motion):
    dof = motion['dof']
    root_qua = motion['root_rot']
    dof_new = np.concatenate((dof[:, :19], dof[:, 22:26]), axis=1)
    root_aa = sRot.from_quat(root_qua).as_rotvec()

    dof_axis = np.load('../description/robots/g1/dof_axis.npy', allow_pickle=True)
    dof_axis = dof_axis.astype(np.float32)

    pose_aa = np.concatenate(
        (np.expand_dims(root_aa, axis=1), dof_axis * np.expand_dims(dof_new, axis=2), np.zeros((dof_new.shape[0], 3, 3))),
        axis=1).astype(np.float32)
    
    return pose_aa,dof_new

def process_motion(motion, cfg):
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot)  # load forward kinematics model

    # breakpoint()

    if 'pose_aa' not in motion.keys():
        pose_aa,dof = count_pose_aa(motion=motion)
        motion['pose_aa'] = pose_aa
        motion['dof'] = dof
        pose_aa = torch.from_numpy(pose_aa).unsqueeze(0)
    else:
        pose_aa = torch.from_numpy(motion['pose_aa']).unsqueeze(0)
    root_trans = torch.from_numpy(motion['root_trans_offset']).unsqueeze(0)

    fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)

    feet_l, feet_r = foot_detect(fk_return.global_translation_extend[0])

    motion['contact_mask'] = np.concatenate([feet_l,feet_r],axis=-1)
    motion['smpl_joints'] = fk_return.global_translation_extend[0].detach().numpy()

    return motion


@hydra.main(version_base=None, config_path="../description/robots/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    folder_path = cfg.input_folder
    if folder_path[-1]=='/':
        target_folder_path = folder_path[:-1] + '_contact_mask'
    else:
        target_folder_path = folder_path+'_contact_mask'
    os.makedirs(target_folder_path, exist_ok=True)
    target_folder_list = os.listdir(target_folder_path)
    for filename in os.listdir(folder_path):
        if filename.split('.')[0] + '_cont_mask.pkl' in target_folder_list:
            continue
        filename = filename.split('.')[0]
        motion_file = folder_path + '/' + f'{filename}.pkl'
        motion_data = joblib.load(motion_file)
        motion_data_keys = list(motion_data.keys())
        motion = process_motion(motion_data[motion_data_keys[0]], cfg)
        save_data={}
        save_data[motion_data_keys[0]] = motion
        dumped_file = f'{target_folder_path}/{filename}_cont_mask.pkl'
        joblib.dump(save_data, dumped_file)

if __name__ == "__main__":
    main()