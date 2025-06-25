import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

import joblib
import torch
import torch.nn.functional as F
import math
from smpl_sim.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, \
    SMPLH_MUJOCO_NAMES
from motion_source.utils.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.utils.smoothing_utils import gaussian_kernel_1d, gaussian_filter_1d_batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf

import motion_source.utils.rotation_conversions as tRot


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in entry_data:
        return
    framerate = entry_data['mocap_framerate']

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate
    }

def foot_detect(positions, thres=0.002):
    fid_r, fid_l = [8, 11], [7, 10]
    positions = positions.numpy()
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.15, 0.1]) 
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:,fid_l,2]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(int) & (feet_l_h < heightfactor).astype(int)).astype(np.float32)
    feet_l = np.concatenate([np.array([[1., 1.]]),feet_l],axis=0)
    feet_l = np.max(feet_l, axis=1, keepdims=True)
    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:,fid_r,2]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(int) & (feet_r_h < heightfactor).astype(int)).astype(np.float32)
    feet_r = np.concatenate([np.array([[1., 1.]]),feet_r],axis=0)
    feet_r = np.max(feet_r, axis=1, keepdims=True)
    return feet_l, feet_r

def process_motion(key_names, key_name_to_pkls, cfg):
    device = torch.device("cpu")

    humanoid_fk = Humanoid_Batch(cfg.robot)  # load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="./smpl_model/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"./retargeted_motion_data/phc/shape_optimized_v1.pkl")

    all_data = {}
    pbar = tqdm(key_names, position=0, leave=True)
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None: continue
        skip = int(amass_data['fps'] // 30)
        trans = torch.from_numpy(amass_data['trans'][::skip])
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float()

        if N < 10:
            print("to short")
            continue

        # import ipdb;ipdb.set_trace()

        # breakpoint()

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            origin_verts, origin_global_trans = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.from_numpy(amass_data['betas']).unsqueeze(0), trans)
            origin_global_trans[..., 2] -= origin_verts[0, :, 2].min().item()
            feet_l , feet_r = foot_detect(origin_global_trans)
            contact_mask = np.concatenate([feet_l,feet_r],axis=-1)

            default_verts, default_joints = smpl_parser_n.get_joints_verts(
                pose_aa_walk, shape_new * 0, trans
            )
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos

        joints[..., 2] -= verts[0, :, 2].min().item()
        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        default_min_joint_each_frame = default_joints[..., 2].min(dim=-1).values
        default_min_joint_each_frame -= default_min_joint_each_frame.min()

        gt_root_rot_quat = torch.from_numpy((sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat(
            [0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float()  # can't directly use this
        gt_root_rot = torch.from_numpy(sRot.from_quat(
            torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float()  # so only use the heading.

        # def dof_to_pose_aa(dof_pos):
        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

        dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)
        optimizer_root = torch.optim.Adam([root_rot_new, root_pos_offset], lr=0.01)

        kernel_size = 5  # Size of the Gaussian kernel
        sigma = 0.75  # Standard deviation of the Gaussian kernel
        B, T, J, D = dof_pos_new.shape

        for iteration in range(cfg.get("fitting_iterations", 1000)):
            # breakpoint()
            pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new,
                                        torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis=2)
            fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None,] + root_pos_offset)

            pose_aa_walk = (pose_aa_walk).reshape(-1, 24, 3)
            if num_augment_joint > 0:
                diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]

            loss_g = diff.norm(dim=-1).mean()
            loss = loss_g

            optimizer_pose.zero_grad()
            optimizer_root.zero_grad()
            loss.backward()
            optimizer_pose.step()
            optimizer_root.step()

            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

            pbar.set_description_str(f"{data_key}-Iter: {iteration} \t {loss.item() * 1000:.3f}")
            dof_pos_new.data = \
                gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None,], kernel_size, sigma).transpose(2,
                                                                                                                     1)[
                    ..., None]

        # import ipdb;ipdb.set_trace()
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new,
                                    torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis=2)

        # import ipdb;ipdb.set_trace()

        height_diff = fk_return.global_translation[..., 2].min().item()
        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

        combined_mesh = humanoid_fk.mesh_fk(pose_aa_h1_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach())
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()

        root_trans_offset_dump[..., 2] -= height_diff

        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff

        data_dump = {
            "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().numpy(),
            "root_rot": sRot.from_rotvec(root_rot_new.detach().numpy()).as_quat(),
            "smpl_joints": joints_dump,
            "fps": 30,
            "contact_mask": contact_mask
        }
        all_data[data_key] = data_dump
    return all_data


@hydra.main(version_base=None, config_path="../../description/robots/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    all_pkls = glob.glob("./motion_data/dataset/*.npz", recursive=True)
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[3:]).replace(".npz", ""): data_path for data_path in
                        all_pkls}
    key_names = ["0-" + "_".join(data_path.split("/")[3:]).replace(".npz", "") for data_path in all_pkls]
    if not cfg.get("fit_all", False):
        key_names = ["0-motion"]

    from multiprocessing import Pool
    jobs = key_names
    num_jobs = 30
    chunk = np.ceil(len(jobs) / num_jobs).astype(int)
    jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]
    if len(job_args) == 1:
        all_data = process_motion(key_names, key_name_to_pkls, cfg)
    else:
        try:
            pool = Pool(num_jobs)  # multi-processing
            all_data_list = pool.starmap(process_motion, job_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        all_data = {}
        for data_dict in all_data_list:
            all_data.update(data_dict)
    # import ipdb; ipdb.set_trace()
    if len(all_data) == 1:
        data_key = list(all_data.keys())[0]
        os.makedirs(f"./retargeted_motion_data/phc/g1", exist_ok=True)
        dumped_file = f"./retargeted_motion_data/phc/g1/{data_key}_origin.pkl"
        print(dumped_file)
        joblib.dump(all_data, dumped_file)
    else:
        os.makedirs(f"./retargeted_motion_data/phc/g1/", exist_ok=True)
        joblib.dump(all_data, f"./retargeted_motion_data/phc/g1/amass_all.pkl")


if __name__ == "__main__":
    main()
