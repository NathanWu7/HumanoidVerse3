"""
motion filter:
1. compute CoM
2. compute CoP
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
from part_volumes import PartVolume
import pickle as pkl
import numpy as np
import torch.nn as nn
from mesh_utils import HDfier
from model import SMPL
import argparse
from scipy.spatial.transform import Rotation as sRot

import matplotlib.pyplot as plt

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

SMPL_PART_BOUNDS = 'data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_segments_bounds.pkl'
FID_TO_PART = 'data/essentials/yogi_segments/smpl/part_meshes_ply/fid_to_part.pkl'
PART_VID_FID = 'data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_part_vid_fid.pkl'
HD_SMPL_MAP = 'data/essentials/hd_model/smpl/smpl_neutral_hd_sample_from_mesh_out.pkl'
SMPL_MODEL_DIR = 'data/smpl'

def recover_rot(motion):
    body_pose = motion['poses']
    trans = motion['trans']
    transform1 = sRot.from_euler('xyz', np.array([np.pi / 2, 0, np.pi]), degrees=False)
    current_global_orient = body_pose[:, :3]
    current_rot = sRot.from_rotvec(current_global_orient)

    transform_inv = transform1.inv()
    original_global_rot = transform_inv * current_rot

    body_pose[:, :3] = original_global_rot.as_rotvec()
    transform_matrix = transform1.as_matrix()

    trans = trans @ transform_matrix

    return body_pose, trans

def smpl_paraser(motion_file):
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    root_trans = motion_file['trans']
    pose_aa = np.concatenate([motion_file['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = motion_file['betas']
    gender = motion_file['gender']  
    origin_verts, origin_global_trans = smpl_parser_n.get_joints_verts(torch.from_numpy(pose_aa), torch.from_numpy(betas).unsqueeze(0), torch.from_numpy(root_trans))
    return origin_verts

def check_conditions(data, epsilon_stab=0.1, epsilon_stab_N=100):
    n = len(data)

    if data[0] >= epsilon_stab or data[-1] >= epsilon_stab:
        return False
    
    if n > epsilon_stab_N:
        indices = [i for i, val in enumerate(data) if val < epsilon_stab]
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] >= epsilon_stab_N:
                return False
            
    return True

class MotionFilter(nn.Module):
    def __init__(self, faces, cop_w, cop_k, contact_thresh, model_type='smpl', device='cuda'):
        super().__init__()
        if model_type == 'smpl':
            num_faces = 13776
            num_verts_hd = 20000

        assert faces is not None, 'Faces tensor is none'

        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long).to(device)
        self.register_buffer('faces', faces)

        with open(SMPL_PART_BOUNDS, 'rb') as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}

        with open(PART_VID_FID, 'rb') as f:
            self.part_vid_fid = pkl.load(f)

        with open(SMPL_PART_BOUNDS, 'rb') as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}
        self.part_order = sorted(self.part_bounds)

        # mapping between vid_hd and fid
        with open(HD_SMPL_MAP, 'rb') as f:
            faces_vert_is_sampled_from = pkl.load(f)['faces_vert_is_sampled_from']
        index_row_col = torch.stack(
            [torch.LongTensor(np.arange(0, num_verts_hd)), torch.LongTensor(faces_vert_is_sampled_from)], dim=0)
        values = torch.ones(num_verts_hd, dtype=torch.float)
        size = torch.Size([num_verts_hd, num_faces])
        hd_vert_on_fid = torch.sparse_coo_tensor(index_row_col, values, size)

        # mapping between fid and part label
        with open(FID_TO_PART, 'rb') as f:
            fid_to_part_dict = pkl.load(f)
        fid_to_part = torch.zeros([len(fid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32)
        for fid, partname in fid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            fid_to_part[fid, part_idx] = 1.

        self.cop_w = cop_w
        self.cop_k = cop_k
        self.contact_thresh = contact_thresh

        self.hdfy_op = HDfier(model_type=model_type)

        self.hd_vid_in_part = self.vertex_id_to_part_mapping(hd_vert_on_fid, fid_to_part)

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part

    def in_hull(self, points, x):
        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T, np.ones((1, n_points))]
        b = np.r_[x, np.ones(1)]
        try:
            lp = linprog(c, A_eq=A, b_eq=b, method='interior-point')
            return lp.success
        except:
            print('Linprog failed. Problem is infeasible')
            return False

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_name, bound_vids in part_bounds.items():
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())
        return torch.vstack(part_volume).permute(1, 0).to(vertices.device)

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol
    
    def compute_diff(self,com,cop):
        com_xz = torch.stack([com[:, 0], torch.zeros_like(com)[:, 0], com[:, 2]], dim=1)
        contact_centroid_xz = torch.stack([cop[:, 0], torch.zeros_like(cop)[:, 0], cop[:, 2]], dim=1)
        diff = (torch.norm(com_xz - contact_centroid_xz, dim=1))
        return diff

    def compute_com_cop(self, vertices):

        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)

        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 1] - ground_plane_height)
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (1 - self.cop_k * vertex_height) + outside_mask * torch.exp(
            -self.cop_w * vertex_height)
        cop = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (
                torch.sum(pressure_weights, dim=1, keepdim=True) + eps)

        return com, cop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--convert_rot', type=bool, default=False)
    args = parser.parse_args()
    folder = args.folder
    convert_rot = args.convert_rot

    batch_size = 1
    device = "cuda:0"

    smpl = SMPL(SMPL_MODEL_DIR,
                batch_size=batch_size,
                create_transl=False).to(device)

    motion_filter = MotionFilter(
        faces=smpl.faces,
        cop_w=10.,
        cop_k=100.,
        contact_thresh=0.1,
        model_type='smpl',
        device=device
    )

    # breakpoint()
    for filename in os.listdir(folder):
        if filename.split('.')[-1] != 'npz':
            continue
        filepath = folder + filename
        motion_file = np.load(filepath, allow_pickle=True)

        if motion_file['trans'].shape[0] <= 100:
            print(f"{filename}: too short!")
            continue

        if convert_rot:
            poses, transl = recover_rot(motion_file)
            transl = torch.from_numpy(transl).to(device)
        else:
            transl = torch.from_numpy(motion_file['trans']).to(device)
            poses = motion_file['poses']

        betas = motion_file['betas']
        betas = torch.from_numpy(betas[:10]).to(torch.float32).to(device)

        if poses.shape[1]==66:
            poses = np.concatenate((poses,np.zeros((poses.shape[0],6))),axis=1)
        else:
            poses = poses[:,:72]
        poses_mat = torch.from_numpy(sRot.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(-1,24,3,3)).to(torch.float32).to(device)

        height_min = 0

        data = []
        for i in range(poses_mat.shape[0]):
            pred_output_world = smpl(betas=betas.unsqueeze(0),
                                    body_pose=poses_mat[i][1:].unsqueeze(0),
                                    transl=transl[i].unsqueeze(0),
                                    global_orient=poses_mat[i][0].unsqueeze(0).unsqueeze(1),
                                    pose2rot=False)
            vertices = pred_output_world.vertices

            if i==0:
                height_min = vertices[0][:,1].min()
            vertices[0][:,1] -= height_min

            com, cop = motion_filter.compute_com_cop(vertices)
            diff = motion_filter.compute_diff(com,cop)
            data.append(diff.item())

        is_filtered = check_conditions(data)
        print(f"{filename}: {is_filtered}")

        