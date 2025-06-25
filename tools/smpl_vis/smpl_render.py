import cv2
import torch
from einops import einsum, rearrange
from utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay, create_camera_sensor, get_video_lwh, get_writer
from utils.smpl_utils import make_smplx
from tqdm import tqdm
import os
import argparse
import numpy as np
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot

from utils.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points

CRF = 23  # 17 is lossless, every +6 halves the mp4 size

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

def to_cuda(data):
    """Move data in the batch to cuda(), carefully handle data that is not tensor"""
    if isinstance(data, torch.Tensor):
        return data.cuda()
    elif isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v) for v in data]
    else:
        return data

def render(motion, filename):
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("./body_model/smplx2smpl_sparse.pt").cuda()
    J_regressor = torch.load("./body_model/smpl_neutral_J_regressor.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    smplx_out = smplx(**to_cuda(motion))

    N = motion['body_pose'].shape[0]

    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=15,    # 20
        target_center_height=1,   # 1.0
        vec_rot=1,
        device='cuda:0',
    )
    
    length, width, height = (N, 1280, 720)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    debug_cam = False

    render_length = length if not debug_cam else 8
    # -- rendering code -- #
    video_path = "./video_output/"
    os.makedirs(video_path, exist_ok=True)
    writer = get_writer(video_path + filename + '_render.mp4', fps=30, crf=CRF)

    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()

def format_motion(motion_file,fps=30):
    N = motion_file['poses'].shape[0]
    skip = int(motion_file['mocap_framerate'] // fps)
    rot, transl = recover_rot(motion_file)
    body_pose = rot[:, 3:66]
    betas = np.tile(motion_file['betas'][:10],(N, 1))
    global_orient = rot[:, :3]
    return {
        'body_pose': torch.from_numpy(body_pose[::skip]).float(),
        'betas': torch.from_numpy(betas[::skip]).float(),
        'global_orient': torch.from_numpy(global_orient[::skip]).float(),
        'transl': torch.from_numpy(transl[::skip]).float()
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    args = parser.parse_args()
    filepath = args.filepath

    device = 'cuda:0'

    filename = filepath.split('/')[-1]
    assert filename.split('.')[-1] == 'npz', "must npz file"

    motion_file = np.load(filepath, allow_pickle = True)
    motion = format_motion(motion_file)
    render(motion, filename.split('.')[0])
