import argparse
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except Exception as e:  # pragma: no cover
    mujoco = None
    _MUJOCO_IMPORT_ERR = e

from scipy.spatial.transform import Rotation as sRot
from .common import parse_motor_axes


    


def axis_angle_to_dof(pose_aa: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Convert per-DOF axis-angle vectors to scalar angles using provided unit axes.

    pose_aa shape: (T, D, 3) for joints (excluding root).
    axes shape: (D, 3) in the same order.
    Output: (T, D) angles in radians (signed along axis directions).
    """
    # project each vector onto its axis: angle = dot(v, axis)
    # broadcasting: (T,D,3) · (D,3) -> (T,D)
    return (pose_aa * axes[None, :, :]).sum(axis=-1)


def key_callback(key):  # minimal controls
    global _paused, _rewind, _speed
    try:
        c = chr(key)
    except Exception:
        c = None
    if key == 27:  # ESC
        os._exit(0)
    if c == ' ':
        _paused = not _paused
    elif c == 'L':
        _speed *= 1.5
    elif c == 'K':
        _speed /= 1.5
    elif c == 'J':
        _rewind = not _rewind


def main():
    parser = argparse.ArgumentParser(description="Visualize a converted MotionLib file (single motion dict)")
    parser.add_argument("--motion", required=True, help="Path to converted .pkl (joblib) with fps/root_trans_offset/pose_aa")
    parser.add_argument("--xml", required=True, help="Robot MJCF XML used for visualization")
    parser.add_argument("--dt", type=float, default=None, help="Override timestep (seconds). Defaults to 1/fps")
    args = parser.parse_args()

    if mujoco is None:  # pragma: no cover
        raise RuntimeError(f"Failed to import mujoco: {_MUJOCO_IMPORT_ERR}")

    motion_path = Path(args.motion)
    motion = joblib.load(motion_path)
    # Accept either a single-motion dict, or a dict-of-one-motion {name: motion}
    if not isinstance(motion, dict):
        raise ValueError("Loaded object is not a dict. Pass a converted .pkl file.")
    if 'pose_aa' not in motion:
        if len(motion) == 1 and isinstance(next(iter(motion.values())), dict):
            motion = next(iter(motion.values()))
        else:
            raise ValueError("Input must be a single-motion dict with keys including 'pose_aa' and 'root_trans_offset', or a dict-of-one-motion {name: motion}.")

    fps = int(motion.get('fps', 30))
    dt = args.dt if args.dt is not None else 1.0 / max(fps, 1)

    root_trans = np.asarray(motion['root_trans_offset'], dtype=np.float32)  # (T,3)
    pose_aa = np.asarray(motion['pose_aa'], dtype=np.float32)               # (T,1+D,3)
    if pose_aa.ndim != 3 or pose_aa.shape[1] < 2 or pose_aa.shape[2] != 3:
        raise ValueError("pose_aa must have shape (T, 1+D, 3)")

    T = pose_aa.shape[0]
    root_rotvec = pose_aa[:, 0, :]                                          # (T,3)
    joint_aa = pose_aa[:, 1:, :]                                            # (T,D,3)

    # Convert root rotvec to XYZW quaternion, then to MuJoCo WXYZ
    root_quat_xyzw = sRot.from_rotvec(root_rotvec).as_quat().astype(np.float32)
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    # Parse axes for joints (in actuator order)
    motor_names, axes = parse_motor_axes(args.xml)
    D_xml = axes.shape[0]
    # Allow extra trailing joints (e.g., 3 extended bodies with zero AA) by slicing
    if joint_aa.shape[1] < D_xml:
        raise ValueError(f"Joint count mismatch: pose_aa has D={joint_aa.shape[1]} < XML motors {D_xml}")
    joint_aa_use = joint_aa[:, :D_xml, :]
    dof = axis_angle_to_dof(joint_aa_use, axes)                              # (T,D_xml)

    # Launch MuJoCo viewer
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    global _paused, _rewind, _speed
    _paused, _rewind, _speed = False, False, 1.0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30

        t = 0
        while viewer.is_running():
            start = time.time()
            idx = int(np.clip(t / dt, 0, T - 1))

            data.qpos[:3] = root_trans[idx]
            data.qpos[3:7] = root_quat_wxyz[idx]
            data.qpos[7:7 + D_xml] = dof[idx]

            mujoco.mj_forward(model, data)
            if not _paused:
                t += dt * (_speed if not _rewind else -_speed)
                if t < 0:
                    t += T * dt
                elif t >= T * dt:
                    t -= T * dt

            viewer.sync()
            remain = model.opt.timestep - (time.time() - start)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
