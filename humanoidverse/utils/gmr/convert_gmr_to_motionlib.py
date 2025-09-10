import argparse
import os
import pickle
from pathlib import Path
from typing import List

import joblib
import numpy as np
from easydict import EasyDict
from .common import (
    parse_motor_axes,
    build_pose_aa,
    insert_interp_segments,
    default_pose_for_g1,
)


def _lower_dof_interpolation(*args, **kwargs):
    raise NotImplementedError("Lower-limb special interpolation has been removed to simplify the pipeline.")


def _slerp_with_fixed_yaw(*args, **kwargs):  # legacy stub to avoid breaking imports
    raise NotImplementedError("Yaw-fixed slerp has been removed; use standard slerp via common.insert_interp_segments.")


def _default_pose_for_g1(D: int):  # kept for backward compat; delegate to common
    return default_pose_for_g1(D)


def _compute_contact_and_joints(xml_path: str,
                                pose_aa: np.ndarray,
                                root_trans: np.ndarray,
                                fps: int):
    """Compute contact_mask (T,2) and smpl_joints (T,NB,3) via FK, mimicking MotionHub.

    Uses Humanoid_Batch FK on the given MJCF. Feet are detected by low velocity and low height.
    """
    try:
        import torch
        from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
        xml_path = str(xml_path)
        asset_root = str(Path(xml_path).parent)
        asset_file = str(Path(xml_path).name)
        cfg = EasyDict({
            'asset': EasyDict({'assetRoot': asset_root, 'assetFileName': asset_file}),
            'extend_config': [],
        })
        fk = Humanoid_Batch(cfg)
        pose_t = torch.from_numpy(pose_aa.astype(np.float32))[None, ...]
        trans_t = torch.from_numpy(root_trans.astype(np.float32))[None, ...]
        fk_ret = fk.fk_batch(pose_t, trans_t, return_full=True, dt=1.0 / max(int(fps), 1))

        if 'global_translation_extend' in fk_ret.__dict__:
            joints = fk_ret.global_translation_extend[0]
        else:
            joints = fk_ret.global_translation[0]
        names = getattr(fk, 'body_names_augment', getattr(fk, 'body_names', None))
        if names is None:
            raise RuntimeError('FK body names unavailable')
        def _find(name_candidates):
            for cand in name_candidates:
                if cand in names:
                    return names.index(cand)
            # fallback: fuzzy
            for i, n in enumerate(names):
                for cand in name_candidates:
                    if cand in n:
                        return i
            raise KeyError(f'Body name not found: {name_candidates}')

        l_idx = _find(['left_ankle_roll_link', 'l_foot', 'left_ankle'])
        r_idx = _find(['right_ankle_roll_link', 'r_foot', 'right_ankle'])

        pos = joints  # (T, NB, 3)
        # velocity-based contact with height threshold
        vel_l = (pos[1:, l_idx] - pos[:-1, l_idx])
        vel_r = (pos[1:, r_idx] - pos[:-1, r_idx])
        speed2_l = (vel_l * vel_l).sum(dim=-1)
        speed2_r = (vel_r * vel_r).sum(dim=-1)
        h_l = pos[1:, l_idx, 2]
        h_r = pos[1:, r_idx, 2]
        velfactor = 0.002
        heightfactor = 0.12
        feet_l = ((speed2_l < velfactor) & (h_l < heightfactor)).float()
        feet_r = ((speed2_r < velfactor) & (h_r < heightfactor)).float()
        # prepend first frame as contact for alignment (same as MotionHub)
        feet_l = torch.cat([torch.ones(1, dtype=feet_l.dtype), feet_l], dim=0)
        feet_r = torch.cat([torch.ones(1, dtype=feet_r.dtype), feet_r], dim=0)
        contact = torch.stack([feet_l, feet_r], dim=-1).cpu().numpy().astype(np.float32)  # (T,2)
        smpl_joints = joints.cpu().numpy().astype(np.float32)
        return smpl_joints, contact
    except Exception as e:
        print(f"[convert] Contact/joints computation failed ({e}); fallback to defaults.")
        T = pose_aa.shape[0]
        NB = pose_aa.shape[1]
        return np.zeros((T, NB, 3), dtype=np.float32), np.full((T, 2), 0.5, dtype=np.float32)


def convert_file(src_path: Path, dst_path: Path, dof_axis: np.ndarray,
                 target_motor_names: List[str],
                 src_motor_names: List[str] = None,
                 interp_start: int = 0,
                 interp_end: int = 0,
                 root_rot_policy: str = "freeze",
                 root_pos_policy: str = "linear",
                 xml_path: str = None) -> None:
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    required = ["fps", "root_pos", "root_rot", "dof_pos"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {src_path}")

    fps = int(data["fps"]) if not isinstance(data["fps"], (list, tuple, np.ndarray)) else int(np.array(data["fps"]).item())
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)        # (T,3)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)   # (T,4) XYZW
    dof_pos_src = np.asarray(data["dof_pos"], dtype=np.float32)       # (T,D_src)

    # Align dof_pos to target motor order when needed
    if dof_axis.shape[0] != dof_pos_src.shape[1]:
        if src_motor_names is None:
            # Heuristic: common G1 29->23 subset mapping (drop wrists)
            # Indices: keep [0..18] + [22..25] from 29-DoF ordering
            Dsrc = dof_pos_src.shape[1]
            if Dsrc >= 26 and dof_axis.shape[0] == 23:
                sel = list(range(19)) + list(range(22, 26))
                dof_pos = dof_pos_src[:, sel]
            else:
                raise RuntimeError(
                    "DOF mismatch and no --src-xml provided. "
                    f"dof_pos has {Dsrc}, target axes {dof_axis.shape[0]}. "
                    "Provide --src-xml so we can align by joint names."
                )
        else:
            # Name-based alignment
            name_to_src_idx = {n: i for i, n in enumerate(src_motor_names)}
            aligned = []
            missing = []
            for n in target_motor_names:
                if n in name_to_src_idx:
                    aligned.append(dof_pos_src[:, name_to_src_idx[n]][:, None])
                else:
                    # pad zeros if missing
                    aligned.append(np.zeros((dof_pos_src.shape[0], 1), dtype=np.float32))
                    missing.append(n)
            if missing:
                print(f"[convert] Warning: missing joints in src: {missing}")
            dof_pos = np.concatenate(aligned, axis=1)
    else:
        dof_pos = dof_pos_src

    # Interpolation (optional): linear for pos/dof, slerp for root rot
    if interp_start > 0 or interp_end > 0:
        def_rt, def_rq, def_dof = _default_pose_for_g1(dof_pos.shape[1])
        root_pos, root_rot_xyzw, dof_pos = insert_interp_segments(
            root_pos, root_rot_xyzw, dof_pos,
            start_frames=interp_start,
            end_frames=interp_end,
            default_root_pos=def_rt,
            default_root_quat_xyzw=def_rq,
            default_dof=def_dof,
            root_rot_policy=root_rot_policy,
            root_pos_policy=root_pos_policy,
        )

    pose_aa = build_pose_aa(root_rot_xyzw, dof_pos, dof_axis)        # (T,1+D,3)
    # For motion_tracking pipeline we use 3 extended bodies (hands, head) without extra DOFs.
    # Append zero AA for them to match Humanoid_Batch's augmented skeleton length.
    EXTEND_BODIES = 3
    if EXTEND_BODIES > 0:
        zeros_ext = np.zeros((pose_aa.shape[0], EXTEND_BODIES, 3), dtype=np.float32)
        pose_aa = np.concatenate([pose_aa, zeros_ext], axis=1)

    # Optional fields to mirror MotionHub output for drop-in compatibility
    smpl_joints = np.zeros_like(pose_aa, dtype=np.float32)
    contact_mask = np.full((pose_aa.shape[0], 2), 0.5, dtype=np.float32)

    if xml_path is not None:
        smpl_joints, contact_mask = _compute_contact_and_joints(xml_path=xml_path,
                                                                pose_aa=pose_aa,
                                                                root_trans=root_pos,
                                                                fps=fps)
    else:
        smpl_joints = np.zeros_like(pose_aa, dtype=np.float32)
        contact_mask = np.full((pose_aa.shape[0], 2), 0.5, dtype=np.float32)
    motion = {
        "fps": fps,
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof_pos.astype(np.float32),
        "root_rot": root_rot_xyzw.astype(np.float32),
        "smpl_joints": smpl_joints,
        "contact_mask": contact_mask,
    }
    # Save a companion .npy (qpos-like): [root_trans (3), root_rot XYZW (4), dof (D)]
    try:
        qpos_like = np.concatenate([root_pos.astype(np.float32),
                                    root_rot_xyzw.astype(np.float32),
                                    dof_pos.astype(np.float32)], axis=1)
        npy_path = dst_path.with_suffix('.npy')
        os.makedirs(npy_path.parent, exist_ok=True)
        np.save(npy_path, qpos_like)
    except Exception as e:
        print(f"[convert] Warning: failed to save npy alongside pkl: {e}")

    key = dst_path.stem
    wrapped = {key: motion}

    os.makedirs(dst_path.parent, exist_ok=True)
    joblib.dump(wrapped, dst_path)


def convert_wrapped_single(src_path: Path, out_file: Path, dof_axis: np.ndarray,
                           target_motor_names: List[str], src_motor_names: List[str] = None,
                           interp_start: int = 0, interp_end: int = 0,
                           root_rot_policy: str = "freeze",
                           root_pos_policy: str = "linear",
                           xml_path: str = None) -> None:
    """Convert one GMR file and write a dict-of-one-motion joblib for training.
    Includes extra keys 'dof' and 'root_rot' to satisfy preprocessing/vis tools.
    """
    with open(src_path, "rb") as f:
        data = pickle.load(f)
    for k in ["fps", "root_pos", "root_rot", "dof_pos"]:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {src_path}")

    fps = int(data["fps"]) if not isinstance(data["fps"], (list, tuple, np.ndarray)) else int(np.array(data["fps"]).item())
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos_src = np.asarray(data["dof_pos"], dtype=np.float32)

    # Align to target motor order
    if dof_axis.shape[0] != dof_pos_src.shape[1]:
        if src_motor_names is None:
            Dsrc = dof_pos_src.shape[1]
            if Dsrc >= 26 and dof_axis.shape[0] == 23:
                sel = list(range(19)) + list(range(22, 26))
                dof_pos = dof_pos_src[:, sel]
            else:
                raise RuntimeError("DOF mismatch and no --src-xml provided for name-based alignment.")
        else:
            name_to_src_idx = {n: i for i, n in enumerate(src_motor_names)}
            cols = []
            for n in target_motor_names:
                cols.append(dof_pos_src[:, name_to_src_idx[n]][:, None] if n in name_to_src_idx else np.zeros((dof_pos_src.shape[0], 1), dtype=np.float32))
            dof_pos = np.concatenate(cols, axis=1)
    else:
        dof_pos = dof_pos_src

    # Interpolation (simple linear/slerp)
    if interp_start > 0 or interp_end > 0:
        def_rt, def_rq, def_dof = _default_pose_for_g1(dof_pos.shape[1])
        root_pos, root_rot_xyzw, dof_pos = insert_interp_segments(
            root_pos, root_rot_xyzw, dof_pos,
            start_frames=interp_start,
            end_frames=interp_end,
            default_root_pos=def_rt,
            default_root_quat_xyzw=def_rq,
            default_dof=def_dof,
            root_rot_policy=root_rot_policy,
            root_pos_policy=root_pos_policy,
        )

    pose_aa = build_pose_aa(root_rot_xyzw, dof_pos, dof_axis)
    EXTEND_BODIES = 3
    if EXTEND_BODIES > 0:
        zeros_ext = np.zeros((pose_aa.shape[0], EXTEND_BODIES, 3), dtype=np.float32)
        pose_aa = np.concatenate([pose_aa, zeros_ext], axis=1)
    if xml_path is not None:
        smpl_joints, contact_mask = _compute_contact_and_joints(xml_path=xml_path,
                                                                pose_aa=pose_aa,
                                                                root_trans=root_pos,
                                                                fps=int(fps))
    else:
        smpl_joints = np.zeros_like(pose_aa, dtype=np.float32)
        contact_mask = np.full((pose_aa.shape[0], 2), 0.5, dtype=np.float32)
    motion = {
        "fps": int(fps),
        "root_trans_offset": root_pos.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof_pos.astype(np.float32),
        "root_rot": root_rot_xyzw.astype(np.float32),
        "smpl_joints": smpl_joints.astype(np.float32),
        "contact_mask": contact_mask.astype(np.float32),
    }
    # Save a companion .npy (qpos-like): [root_trans (3), root_rot XYZW (4), dof (D)]
    try:
        qpos_like = np.concatenate([root_pos.astype(np.float32),
                                    root_rot_xyzw.astype(np.float32),
                                    dof_pos.astype(np.float32)], axis=1)
        npy_path = out_file.with_suffix('.npy')
        os.makedirs(npy_path.parent, exist_ok=True)
        np.save(npy_path, qpos_like)
    except Exception as e:
        print(f"[convert] Warning: failed to save npy alongside pkl: {e}")

    key = out_file.stem
    wrapped = {key: motion}
    os.makedirs(out_file.parent, exist_ok=True)
    joblib.dump(wrapped, out_file)



def main():
    parser = argparse.ArgumentParser(description="Convert GMR outputs to MotionLib format (per-file .pkl)")
    parser.add_argument("--src", type=str, required=True, help="Path to GMR output file or directory")
    parser.add_argument("--xml", type=str, required=True, help="Target robot MJCF xml used by training")
    parser.add_argument("--src-xml", type=str, default=None, help="(Optional) Robot MJCF used to produce GMR dof_pos. Enables name-based alignment.")
    parser.add_argument("--out", type=str, required=True, help="Output file or directory for MotionLib .pkl")
    parser.add_argument("--interp-start", type=int, default=30, help="Frames from default -> first key (fixed policy)")
    parser.add_argument("--interp-end", type=int, default=30, help="Frames from last key -> default (fixed policy)")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    target_motor_names, dof_axis = parse_motor_axes(args.xml)
    src_motor_names = parse_motor_axes(args.src_xml)[0] if args.src_xml else None

    _INTERP_START = int(args.interp_start)
    _INTERP_END = int(args.interp_end)

    # Fixed interpolation policies to match reference script: root_rot freeze, root_pos linear_z
    _ROOT_ROT_POLICY = "freeze"
    _ROOT_POS_POLICY = "linear_z"

    if out.suffix == ".pkl":
        # Produce a dict-of-one-motion file suitable for training
        if src.is_dir():
            # Pick the first motion file
            files = sorted([p for p in src.rglob("*.pkl")])
            if not files:
                raise FileNotFoundError(f"No .pkl found under {src}")
            use_src = files[0]
        else:
            use_src = src
        convert_wrapped_single(use_src, out, dof_axis, target_motor_names, src_motor_names,
                               _INTERP_START, _INTERP_END, _ROOT_ROT_POLICY, _ROOT_POS_POLICY, args.xml)
        print(f"Wrapped single motion -> {out}")
    else:
        # Directory output: mirror structure with per-file MotionLib format
        if not src.is_dir():
            raise ValueError("When --out is a directory, --src must be a directory")
        cnt = 0
        for root, _, files in os.walk(src):
            for fn in files:
                if not fn.endswith(".pkl"):
                    continue
                src_fp = Path(root) / fn
                rel = src_fp.relative_to(src)
                dst_fp = out / rel
                convert_file(src_fp, dst_fp, dof_axis, target_motor_names, src_motor_names,
                             _INTERP_START, _INTERP_END, _ROOT_ROT_POLICY, _ROOT_POS_POLICY, args.xml)
                cnt += 1
        print(f"Converted {cnt} files -> {out}")


if __name__ == "__main__":
    main()
