from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    args = parser.parse_args()
    filepath = args.filepath
    motion_file = np.load(filepath, allow_pickle=True)
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

    root_trans = motion_file['trans']
    pose_aa = np.concatenate([motion_file['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = motion_file['betas']
    gender = motion_file['gender']  

    origin_verts, origin_global_trans = smpl_parser_n.get_joints_verts(torch.from_numpy(pose_aa), torch.from_numpy(betas).unsqueeze(0), torch.from_numpy(root_trans))

    breakpoint()