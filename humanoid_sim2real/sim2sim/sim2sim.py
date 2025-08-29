import os
import sys
import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
from collections import deque

from omegaconf import OmegaConf
from humanoid_sim2real.motion_lib.motion_lib_robot import MotionLibRobot
import onnxruntime
import torch.jit


def load_onnx_policy(path: str):
    """加载 ONNX 格式的策略网络，并返回一个推理函数"""
    session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    def run_inference(input_numpy: np.ndarray) -> torch.Tensor:
        """执行模型前向推理"""
        ort_inputs = {input_name: input_numpy}
        ort_outs = session.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], dtype=torch.float32).to('cpu', copy=True)

    return run_inference


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def get_gravity_orientation(quaternion):
    # quaternion in (w, x, y, z) format
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.array([
        2 * (qx * qz - qw * qy),
        2 * (qy * qz + qw * qx),
        qw * qw - qx * qx - qy * qy + qz * qz
    ])
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom - 1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])


if __name__ == "__main__":
    # 直接使用硬件部署的配置文件，确保完全一致
    hardware_config_path = "humanoid_sim2real/configs/g1_ref_kungfu.yaml"

    with open(hardware_config_path) as file:
        config = OmegaConf.load(file)

        # --- 策略和模型路径 ---
        action_policy_path = config['action_policy_path']
        init_policy_path = config['init_policy_path']  # 自稳/行走策略
        xml_path = "data/robots/g1/g1_29dof_anneal_23dof.xml"

        # --- 仿真参数 ---
        simulation_dt = 0.001
        control_decimation = 20  # 50Hz control frequency
        dt = simulation_dt * control_decimation  # 控制频率对应的时间步长

        # --- 自动演示参数 ---
        locomotion_duration = 5.0  # 自稳策略持续时间（秒）

        # --- 动作策略 (功夫) 的PD参数 (23-DOF) ---
        kps_29dof = np.array(config["kps"], dtype=np.float32)
        kds_29dof = np.array(config["kds"], dtype=np.float32)
        kps = np.concatenate((kps_29dof[:19], kps_29dof[22:26]))
        kds = np.concatenate((kds_29dof[:19], kds_29dof[22:26]))

        # --- 初始策略 (自稳) 的PD参数 (29-DOF) ---
        init_kps = np.array(config["init_kps"], dtype=np.float32)
        init_kds = np.array(config["init_kds"], dtype=np.float32)

        # --- 关节角度 ---
        default_angles_29dof = np.array(config["default_dof_pos"], dtype=np.float32)
        default_angles = np.concatenate((default_angles_29dof[:19], default_angles_29dof[22:26]))
        default_pre_angles_29dof = np.array(config["default_pre_dof_pos"], dtype=np.float32)

        # --- 缩放参数 ---
        ang_vel_scale = config["scale_base_ang_vel"]
        dof_pos_scale = config["scale_dof_pos"]
        dof_vel_scale = config["scale_dof_vel"]
        action_scale = config["scale_actions"]

        # --- 维度信息 ---
        num_actions = config["num_actions"]
        num_obs = config["num_observations"]
        num_history = config["obs_context_len"]
        init_num_actions = config["init_num_actions"]
        init_num_obs = config["init_num_obs"]
        init_num_history = config["init_obs_context_len"]

        # --- Motion Library (用于功夫动作) ---
        motion_lib = MotionLibRobot(config["motion"], num_envs=1, device="cpu")
        motion_lib.load_motions(random_sample=False)
        motion_len = motion_lib.get_motion_length([0])
        motion_dt = motion_lib._motion_dt

        # --- 关节和力矩限制 (23-DOF) ---
        joint_limit_lo_29dof = np.array(config["dof_pos_lower_limit_list"])
        joint_limit_hi_29dof = np.array(config["dof_pos_upper_limit_list"])

        joint_limit_lo = np.concatenate((joint_limit_lo_29dof[:19], joint_limit_lo_29dof[22:26]))
        joint_limit_hi = np.concatenate((joint_limit_hi_29dof[:19], joint_limit_hi_29dof[22:26]))

        soft_dof_pos_limit = 0.95
        for i in range(len(joint_limit_lo)):
            if i != 5 and i != 11 and i != 4 and i != 10:
                m = (joint_limit_lo[i] + joint_limit_hi[i]) / 2
                r = joint_limit_hi[i] - joint_limit_lo[i]
                joint_limit_lo[i] = m - 0.5 * r * soft_dof_pos_limit
                joint_limit_hi[i] = m + 0.5 * r * soft_dof_pos_limit

    # --- 模型和数据加载 ---
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    action_policy = load_onnx_policy(action_policy_path)
    initial_policy = torch.jit.load(init_policy_path)
    initial_policy.to('cpu')

    # --- 自动演示控制标志位 ---
    init_policy_flag = True  # 直接开始自稳策略
    start_policy = False

    # --- 计时器 ---
    start_time = time.time()
    locomotion_start_time = time.time()

    # --- 和实机代码完全一致的变量初始化 ---
    counter = 0
    episode_length_buf = torch.zeros(1, dtype=torch.long)
    motion_start_times = torch.zeros(1, dtype=torch.float32)

    # --- 观测历史缓冲区初始化 ---
    action_hist_obs = deque(maxlen=num_history)
    for _ in range(num_history):
        action_hist_obs.append(np.zeros(num_obs, dtype=np.float32))

    init_obs_history = deque(maxlen=init_num_history)
    for _ in range(init_num_history):
        init_obs_history.append(np.zeros(init_num_obs, dtype=np.float32))

    # --- 动作历史 ---
    prev_action = np.zeros(num_actions, dtype=np.float32)
    init_prev_action = np.zeros(init_num_actions, dtype=np.float32)

    # --- 命令和观测变量 ---
    xyyaw_command = np.array([0., 0., 0.], dtype=np.float32)
    commands_scale = np.array([2.0, 2.0, 0.25])
    up_axis_idx = 2
    gravity_vec = torch.zeros((1, 3), dtype=torch.float32)
    gravity_vec[:, up_axis_idx] = -1

    # --- 步态相关 ---
    cycle_time = 0.64
    gait_indices = torch.zeros(1, dtype=torch.float)

    # --- 历史字典初始化 ---
    hist_dict = {
        "actions": np.zeros(num_actions * (num_history - 1), dtype=np.float32),
        "base_ang_vel": np.zeros(3 * (num_history - 1), dtype=np.float32),
        "dof_pos": np.zeros(num_actions * (num_history - 1), dtype=np.float32),
        "dof_vel": np.zeros(num_actions * (num_history - 1), dtype=np.float32),
        "projected_gravity": np.zeros(3 * (num_history - 1), dtype=np.float32),
        "ref_motion_phase": np.zeros(1 * (num_history - 1), dtype=np.float32),
    }

    # 观测缓冲区
    obs_buf = np.zeros(num_obs * num_history, dtype=np.float32)

    # --- 初始化仿真状态 ---
    d.qpos[7:] = default_pre_angles_29dof[:23]  # 设置为自稳姿态
    d.qvel[:] = 0.0
    target_dof_pos = d.qpos[7:].copy()

    print("=== 自动演示模式 ===")
    print(f"程序将自动运行：")
    print(f"1. 首先执行 {locomotion_duration} 秒的自稳/行走策略")
    print(f"2. 然后执行 {motion_len.item():.1f} 秒的功夫动作")
    print(f"3. 循环重复上述过程")
    print("按 Ctrl+C 退出程序\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 添加可视化元素
        for _ in range(28):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        viewer.user_scn.geoms[27].pos = [0, 0, 0]

        print("开始执行自稳策略...")

        while viewer.is_running():
            step_start = time.time()

            current_time = time.time()

            # --- 自动状态切换逻辑 ---
            if init_policy_flag and not start_policy:
                # 检查是否需要切换到动作策略
                locomotion_elapsed = current_time - locomotion_start_time
                if locomotion_elapsed >= locomotion_duration:
                    start_policy = True
                    episode_length_buf = torch.zeros(1, dtype=torch.long)
                    motion_start_times = torch.zeros(1, dtype=torch.float32)
                    print(f"\n切换到功夫动作策略 (自稳时长: {locomotion_elapsed:.1f}s)")
                else:
                    # 显示自稳倒计时
                    remaining = locomotion_duration - locomotion_elapsed
                    if int(remaining) != int(remaining + dt):  # 每秒显示一次
                        print(f"自稳策略运行中... 剩余 {remaining:.1f}s 切换到动作策略")

            # --- 模拟IMU数据 ---
            quat_wxyz = d.qpos[3:7]  # w, x, y, z
            omega = d.qvel[3:6]
            quat_xyzw = torch.tensor([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                                     dtype=torch.float32).unsqueeze(0)
            projected_gravity = quat_rotate_inverse(quat_xyzw, gravity_vec).squeeze(0).numpy()
            obs_ang_vel = omega * ang_vel_scale

            # --- 关节数据 ---
            joint_pos = d.qpos[7:]
            joint_vel = d.qvel[6:]

            # --- 控制逻辑 (和实机代码完全一致) ---
            if counter % control_decimation == 0:
                qj_23dof = d.qpos[7:]
                dqj_23dof = d.qvel[6:]

                if init_policy_flag and not start_policy:
                    # --- 执行初始化策略 (自稳策略) ---

                    # 步态更新 - 使用和硬件相同的逻辑
                    walking_mask0 = np.abs(xyyaw_command[0]) > 0.1
                    walking_mask1 = np.abs(xyyaw_command[1]) > 0.1
                    walking_mask2 = np.abs(xyyaw_command[2]) > 0.2
                    walking_cmd_mask = walking_mask0 | walking_mask1 | walking_mask2

                    # 检查IMU姿态是否倾斜
                    roll_threshold = abs(projected_gravity[0]) > 0.05  # roll检查
                    pitch_threshold = abs(projected_gravity[1]) > 0.1   # pitch检查
                    imu_tilt_mask = roll_threshold | pitch_threshold

                    standing_mask = not (walking_cmd_mask | imu_tilt_mask)

                    gait_indices = torch.remainder(gait_indices + dt / cycle_time, 1.0)
                    if standing_mask:
                        gait_indices[:] = 0

                    # 相位计算
                    phase = gait_indices
                    sin_pos = torch.sin(2 * torch.pi * phase)
                    cos_pos = torch.cos(2 * torch.pi * phase)

                    # 关节位置和速度观测
                    q_29dof_sim = np.zeros(29, dtype=np.float32)
                    q_29dof_sim[:19] = qj_23dof[:19]
                    q_29dof_sim[22:26] = qj_23dof[19:23]
                    dq_29dof_sim = np.zeros(29, dtype=np.float32)
                    dq_29dof_sim[:19] = dqj_23dof[:19]
                    dq_29dof_sim[22:26] = dqj_23dof[19:23]

                    pre_obs_joint_pos = (q_29dof_sim - default_pre_angles_29dof) * dof_pos_scale
                    pre_obs_joint_vel = dq_29dof_sim * dof_vel_scale

                    # 构建观测 - 添加IMU rpy角度数据，和硬件代码保持一致
                    # 从四元数计算rpy角度
                    qw, qx, qy, qz = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
                    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
                    pitch = np.arcsin(2*(qw*qy - qz*qx))
                    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                    obs_imu = np.array([roll, pitch, yaw]) * 1.0  # scale_project_gravity

                    obs_now = np.concatenate([
                        sin_pos, cos_pos, xyyaw_command * commands_scale,
                        pre_obs_joint_pos[:29], pre_obs_joint_vel[:29], init_prev_action,
                        obs_ang_vel, obs_imu, np.zeros(6)
                    ]).astype(np.float32)

                    init_obs_history.append(obs_now)
                    flat_obs = np.concatenate(list(init_obs_history)).ravel()
                    obs_tensor = torch.from_numpy(flat_obs).unsqueeze(0)

                    # 策略推理
                    with torch.no_grad():
                        action_29dof = initial_policy(obs_tensor).cpu().numpy().squeeze()
                    init_prev_action = action_29dof

                    # 计算目标位置
                    actions_scaled_29dof = action_29dof * action_scale
                    target_dof_pos_29dof = actions_scaled_29dof + default_pre_angles_29dof
                    target_dof_pos = np.concatenate([
                        target_dof_pos_29dof[:19],
                        target_dof_pos_29dof[22:26]
                    ])

                    # 使用初始化PD增益
                    current_kps = np.concatenate([init_kps[:19], init_kps[22:26]])
                    current_kds = np.concatenate([init_kds[:19], init_kds[22:26]])

                elif init_policy_flag and start_policy:
                    # --- 执行动作策略 (功夫动作) ---

                    # 关节观测 - 修复：直接使用23维数据，不需要映射
                    obs_joint_pos = joint_pos - default_angles
                    obs_joint_pos_scaled = obs_joint_pos * dof_pos_scale
                    obs_joint_vel_scaled = joint_vel * dof_vel_scale  # 直接使用，不需要拼接

                    # 运动相位
                    motion_times = (episode_length_buf + 1) * dt + motion_start_times
                    ref_motion_phase = motion_times / motion_len

                    # 构建观测
                    obs_buf[:num_actions] = prev_action.copy()
                    obs_buf[num_actions:num_actions + 3] = obs_ang_vel.copy()
                    obs_buf[num_actions + 3:num_actions * 2 + 3] = obs_joint_pos_scaled.copy()
                    obs_buf[num_actions * 2 + 3:num_actions * 3 + 3] = obs_joint_vel_scaled.copy()

                    # 历史观测
                    history_numpy = []
                    for key in sorted(hist_dict.keys()):
                        history_numpy.append(hist_dict[key])
                    obs_buf[num_actions * 3 + 3:num_actions * 3 + 3 + num_obs * (num_history - 1)] = np.concatenate(
                        history_numpy, axis=-1)
                    obs_buf[num_actions * 3 + 3 + num_obs * (num_history - 1):num_actions * 3 + 6 + num_obs * (
                                num_history - 1)] = projected_gravity
                    obs_buf[num_actions * 3 + 6 + num_obs * (num_history - 1):] = ref_motion_phase.cpu()

                    # 策略推理
                    obs_tensor = obs_buf.reshape(1, -1)
                    action = action_policy(obs_tensor).cpu().numpy().squeeze()
                    prev_action = action

                    # 更新历史
                    hist_dict["actions"] = np.concatenate([prev_action, hist_dict["actions"][:-num_actions]])
                    hist_dict["base_ang_vel"] = np.concatenate([obs_ang_vel, hist_dict["base_ang_vel"][:-3]])
                    hist_dict["dof_pos"] = np.concatenate([obs_joint_pos_scaled, hist_dict["dof_pos"][:-num_actions]])
                    hist_dict["dof_vel"] = np.concatenate([obs_joint_vel_scaled, hist_dict["dof_vel"][:-num_actions]])
                    hist_dict["projected_gravity"] = np.concatenate(
                        [projected_gravity, hist_dict["projected_gravity"][:-3]])
                    hist_dict["ref_motion_phase"] = np.concatenate(
                        [ref_motion_phase.cpu(), hist_dict["ref_motion_phase"][:-1]])

                    # 计算目标位置
                    actions_scaled = action * action_scale
                    target_dof_pos = actions_scaled + default_angles

                    # 使用动作PD增益
                    current_kps = kps
                    current_kds = kds

                    # 更新计数器
                    episode_length_buf += 1

                    # 显示进度条
                    if episode_length_buf % 25 == 0:  # 每0.5秒显示一次 (25 * 0.02s = 0.5s)
                        current_time_motion = episode_length_buf * dt + motion_start_times
                        progress = current_time_motion / motion_len
                        bar_length = 30
                        filled_length = int(bar_length * progress)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        sys.stdout.write(
                            f"\r功夫动作进度: [{bar}] {int(progress * 100)}% ({current_time_motion.item():.1f}s/{motion_len.item():.1f}s)")
                        sys.stdout.flush()

                    # 检查动作是否完成
                    current_time_motion = episode_length_buf * dt + motion_start_times
                    if current_time_motion > motion_len:
                        print(f"\n功夫动作执行完毕 (耗时: {current_time_motion.item():.1f}s)")
                        print("切换回自稳策略...")
                        start_policy = False
                        episode_length_buf = torch.zeros(1, dtype=torch.long)
                        locomotion_start_time = time.time()  # 重置自稳计时器
                else:
                    # --- 默认状态，保持当前位置 ---
                    current_kps = np.concatenate([init_kps[:19], init_kps[22:26]])
                    current_kds = np.concatenate([init_kds[:19], init_kds[22:26]])

            # --- 物理仿真 ---
            tau = pd_control(target_dof_pos, d.qpos[7:], current_kps, np.zeros_like(d.qvel[6:]), d.qvel[6:],
                             current_kds)
            d.ctrl[:] = tau

            mujoco.mj_step(m, d)
            counter += 1

            # --- 可视化参考动作 ---
            if start_policy:
                motion_res_cur = motion_lib.get_motion_state([0], torch.tensor([episode_length_buf * dt]))
                ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]
                for i in range(ref_body_pos_extend.shape[0]):
                    viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i] + torch.tensor([1., 0., 0.])
            else:
                # 隐藏参考可视化
                for i in range(28):
                    viewer.user_scn.geoms[i].pos = [10, 10, 10]

            viewer.sync()

            # --- 安全检查 ---
            if ((d.qpos[7:] - joint_limit_lo) < -0.02).any() or ((d.qpos[7:] - joint_limit_hi) > 0.02).any():
                print("\n关节限制达到，程序停止")
                break

            # --- 控制仿真速度 ---
            time_until_next_step = simulation_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("\n仿真结束")