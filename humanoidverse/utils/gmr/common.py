import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np


def parse_motor_axes(xml_path: str) -> Tuple[List[str], np.ndarray]:
    """Parse motor target joint names and corresponding joint axes from MuJoCo MJCF.

    Returns:
        names: list of joint names in actuator (motor) order
        axes: (D, 3) array of unit axes aligned with each motor's joint
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_by_name: Dict[str, ET.Element] = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name:
            joint_by_name[name] = joint

    actuator = root.find(".//actuator")
    if actuator is None:
        raise ValueError(f"No <actuator> section in {xml_path}")
    motors = actuator.findall("motor")
    if not motors:
        raise ValueError(f"No <motor> in <actuator> for {xml_path}")

    names: List[str] = []
    axes: List[List[float]] = []
    for m in motors:
        jname = m.get("joint")
        if jname is None:
            raise ValueError("Encountered <motor> without a 'joint' attribute")
        names.append(jname)
        j = joint_by_name.get(jname)
        axis = [0.0, 0.0, 0.0]
        if j is not None and j.get("axis"):
            axis = [float(x) for x in j.get("axis").strip().split()]
        axes.append(axis)

    axes_arr = np.asarray(axes, dtype=np.float32)
    # normalize to unit length to make angle = dot(rotvec, axis)
    norms = np.linalg.norm(axes_arr, axis=1, keepdims=True) + 1e-9
    axes_arr = axes_arr / norms
    return names, axes_arr


def build_pose_aa(root_rot_xyzw: np.ndarray, dof_pos: np.ndarray, dof_axis: np.ndarray) -> np.ndarray:
    """Compose axis-angle pose sequence from root quaternion and scalar dof positions.

    Args:
        root_rot_xyzw: (T, 4) quaternion in XYZW
        dof_pos: (T, D) scalar joint angles in radians
        dof_axis: (D, 3) unit axes per DOF

    Returns:
        pose_aa: (T, 1 + D, 3) with root rotvec at index 0 and per-DOF axis-angle vectors afterwards
    """
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError("root_rot_xyzw must have shape (T,4) in XYZW order")
    if dof_pos.ndim != 2 or dof_axis.ndim != 2 or dof_axis.shape[1] != 3:
        raise ValueError("dof_pos must be (T,D) and dof_axis (D,3)")
    T, D = dof_pos.shape
    if dof_axis.shape[0] != D:
        raise ValueError(f"dof_axis length {dof_axis.shape[0]} does not match dof_pos dim {D}")

    from scipy.spatial.transform import Rotation as sRot
    root_rotvec = sRot.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)  # (T,3)
    dof_axis_t = np.broadcast_to(dof_axis[None, :, :], (T, D, 3)).astype(np.float32)
    dof_angles = dof_pos.astype(np.float32)[:, :, None]
    dof_aa = dof_axis_t * dof_angles
    return np.concatenate([root_rotvec[:, None, :], dof_aa], axis=1)


def insert_interp_segments(root_pos: np.ndarray,
                           root_rot_xyzw: np.ndarray,
                           dof_pos: np.ndarray,
                           start_frames: int,
                           end_frames: int,
                           default_root_pos: np.ndarray,
                           default_root_quat_xyzw: np.ndarray,
                           default_dof: np.ndarray,
                           root_rot_policy: str = "freeze",
                           root_pos_policy: str = "linear") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert linear/slerp interpolation segments at the start/end.

    - root_pos: (T,3)
    - root_rot_xyzw: (T,4)
    - dof_pos: (T,D)
    - default_*: default pose used for padding/interp

    Returns new arrays with shape (T + start_frames + end_frames, ...)
    Interpolation policy:
      - root position (root_pos_policy):
          'linear'   -> linear for XYZ between default and boundary keys.
          'freeze'   -> hold boundary XY Z values constant (no interp).
          'linear_z' -> X/Y freeze to boundary; only Z linear between default and boundary.
      - root rotation (root_rot_policy): slerp/keep_yaw/freeze between default and first/last key.
      - dof_pos: linear interpolation per joint.
    """
    T, D = dof_pos.shape
    if start_frames < 0 or end_frames < 0:
        raise ValueError("start_frames/end_frames must be >= 0")
    if start_frames == 0 and end_frames == 0:
        return root_pos.astype(np.float32), root_rot_xyzw.astype(np.float32), dof_pos.astype(np.float32)

    def linspace_exclusive(a, b, n):
        if n <= 0:
            return np.zeros((0,) + a.shape, dtype=np.float32)
        # exclude the first endpoint to avoid duplicate frame with key 0
        return np.linspace(a, b, num=n + 1, dtype=np.float32)[1:]

    def slerp_quat(q0, q1, n):
        if n <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        from scipy.spatial.transform import Rotation as sRot, Slerp
        # enforce short-arc by flipping q1 if needed (quaternion double cover)
        q0 = np.asarray(q0, dtype=np.float32)
        q1 = np.asarray(q1, dtype=np.float32)
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        key_rots = sRot.from_quat(np.stack([q0, q1], axis=0))
        s = Slerp([0, 1], key_rots)
        ts = np.linspace(0, 1, num=n + 1, dtype=np.float32)[1:]
        return s(ts).as_quat().astype(np.float32)

    def slerp_keep_yaw(q_from, q_to, n, yaw_ref):
        if n <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        from scipy.spatial.transform import Rotation as sRot, Slerp
        e_from = sRot.from_quat(q_from).as_euler('ZYX')
        e_to = sRot.from_quat(q_to).as_euler('ZYX')
        e0 = np.array([yaw_ref, e_from[1], e_from[2]], dtype=np.float32)
        e1 = np.array([yaw_ref, e_to[1], e_to[2]], dtype=np.float32)
        key_rots = sRot.from_euler('ZYX', [e0, e1])
        s = Slerp([0, 1], key_rots)
        ts = np.linspace(0, 1, num=n + 1, dtype=np.float32)[1:]
        return s(ts).as_quat().astype(np.float32)

    # start segment
    if start_frames > 0:
        if root_pos_policy == "freeze":
            start_rt = np.tile(root_pos[0], (start_frames, 1)).astype(np.float32)
        elif root_pos_policy == "linear_z":
            start_rt = np.tile(root_pos[0], (start_frames, 1)).astype(np.float32)
            # linear only z from default to first key
            start_rt[:, 2] = linspace_exclusive(np.asarray([default_root_pos[2]], dtype=np.float32),
                                               np.asarray([root_pos[0, 2]], dtype=np.float32),
                                               start_frames)[:, 0]
        else:  # linear
            start_rt = linspace_exclusive(np.asarray(default_root_pos, dtype=np.float32), root_pos[0], start_frames)
        start_dof = linspace_exclusive(np.asarray(default_dof, dtype=np.float32), dof_pos[0], start_frames)
        if root_rot_policy == "slerp":
            start_rq = slerp_quat(np.asarray(default_root_quat_xyzw, dtype=np.float32), root_rot_xyzw[0], start_frames)
        elif root_rot_policy == "keep_yaw":
            from scipy.spatial.transform import Rotation as sRot
            yaw_ref = sRot.from_quat(root_rot_xyzw[0]).as_euler('ZYX')[0]
            start_rq = slerp_keep_yaw(np.asarray(default_root_quat_xyzw, dtype=np.float32), root_rot_xyzw[0], start_frames, yaw_ref)
        else:  # freeze
            start_rq = np.tile(root_rot_xyzw[0], (start_frames, 1)).astype(np.float32)
    else:
        start_rt = np.zeros((0, 3), dtype=np.float32)
        start_rq = np.zeros((0, 4), dtype=np.float32)
        start_dof = np.zeros((0, D), dtype=np.float32)

    # end segment
    if end_frames > 0:
        if root_pos_policy == "freeze":
            end_rt = np.tile(root_pos[-1], (end_frames, 1)).astype(np.float32)
        elif root_pos_policy == "linear_z":
            end_rt = np.tile(root_pos[-1], (end_frames, 1)).astype(np.float32)
            end_rt[:, 2] = linspace_exclusive(np.asarray([root_pos[-1, 2]], dtype=np.float32),
                                              np.asarray([default_root_pos[2]], dtype=np.float32),
                                              end_frames)[:, 0]
        else:  # linear
            end_rt = linspace_exclusive(root_pos[-1], np.asarray(default_root_pos, dtype=np.float32), end_frames)
        end_dof = linspace_exclusive(dof_pos[-1], np.asarray(default_dof, dtype=np.float32), end_frames)
        if root_rot_policy == "slerp":
            end_rq = slerp_quat(root_rot_xyzw[-1], np.asarray(default_root_quat_xyzw, dtype=np.float32), end_frames)
        elif root_rot_policy == "keep_yaw":
            from scipy.spatial.transform import Rotation as sRot
            yaw_ref = sRot.from_quat(root_rot_xyzw[-1]).as_euler('ZYX')[0]
            end_rq = slerp_keep_yaw(root_rot_xyzw[-1], np.asarray(default_root_quat_xyzw, dtype=np.float32), end_frames, yaw_ref)
        else:  # freeze
            end_rq = np.tile(root_rot_xyzw[-1], (end_frames, 1)).astype(np.float32)
    else:
        end_rt = np.zeros((0, 3), dtype=np.float32)
        end_rq = np.zeros((0, 4), dtype=np.float32)
        end_dof = np.zeros((0, D), dtype=np.float32)

    new_root_pos = np.vstack([start_rt, root_pos.astype(np.float32), end_rt])
    new_root_rot = np.vstack([start_rq, root_rot_xyzw.astype(np.float32), end_rq])
    new_dof_pos = np.vstack([start_dof, dof_pos.astype(np.float32), end_dof])
    return new_root_pos, new_root_rot, new_dof_pos


def default_pose_for_g1(D: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a reasonable default root position, orientation, and dof vector for Unitree G1.

    If D != canonical length, slice/pad with zeros to fit.
    """
    root_t = np.array([0.0, 0.0, 0.80], dtype=np.float32)
    root_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # XYZW
    default_dof_29 = np.array([
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
         0.0, 0.0, 0.0,
         0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
         0.2,-0.2, 0.0, 0.9, 0.0, 0.0, 0.0
    ], dtype=np.float32)
    if D == 29:
        return root_t, root_q, default_dof_29
    # 29 -> 23 subset commonly used for G1 (drop wrists): keep [0..18] + [22..25]
    sel = list(range(19)) + list(range(22, 26))
    default_dof_23 = default_dof_29[sel]
    if D == 23:
        return root_t, root_q, default_dof_23
    if D < default_dof_23.shape[0]:
        return root_t, root_q, default_dof_23[:D]
    pad = np.zeros(D - default_dof_23.shape[0], dtype=np.float32)
    return root_t, root_q, np.concatenate([default_dof_23, pad], axis=0)
