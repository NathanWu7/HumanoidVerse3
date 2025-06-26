import os
import argparse

#from utils.crc import CRC

import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

# import rclpy
# from rclpy.node import Node

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_ as MotorCmd
from unitree_sdk2py.utils.crc import CRC
# from unitree_hg.msg import (
#     LowState,
#     MotorState,
#     IMUState,
#     LowCmd,
#     MotorCmd,
# )
import time
from collections import deque

from humanoid_sim2real.common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode

from humanoid_sim2real.common.remote_controller import RemoteController, KeyMap

# from gamepad import Gamepad, parse_remote_data

from tools.motion_lib.motion_lib_robot import MotionLibRobot
from omegaconf import OmegaConf
import sys
import onnxruntime

DEBUG = False
SIM = False

if DEBUG:
    import mujoco
    import mujoco.viewer
    
HW_DOF = 29

USE_WIRELESS_REMOTE = True
LOG_DATA = False
NO_MOTOR = False

HUMANOID_XML = "data/robots/g1/g1_29dof_anneal_23dof.xml"

lowcmd_topic = "rt/lowcmd"
lowstate_topic = "rt/lowstate"


#crc = CRC()

def load_onnx_policy(path: str):
    """
    加载 ONNX 格式的策略网络，并返回一个推理函数
    
    参数：
        path: ONNX 模型文件的绝对路径
    
    返回：
        run_inference: 推理函数，接收 numpy 输入，返回 torch.Tensor 动作输出
    """
    session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    def run_inference(input_numpy: np.ndarray) -> torch.Tensor:
        """
        执行模型前向推理
        
        参数：
            input_numpy: 输入观测, shape: (1, obs_dim), dtype: float32
        
        返回：
            action_tensor: 模型输出, shape: (1, action_dim), device="cuda:0", dtype=torch.float32
        """
        ort_inputs = {input_name: input_numpy}
        ort_outs = session.run(None, ort_inputs)
        # 将 numpy 输出转换为 GPU 上的 torch Tensor
        return torch.tensor(ort_outs[0], dtype=torch.float32).to('cpu', copy=True)
    
    return run_inference


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
    
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class G1():
    def __init__(self,task='stand'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = task

        self.configs = OmegaConf.load("humanoid_sim2real/configs/g1_ref_kungfu.yaml")

        # settings for action policy
        self.num_envs = 1 
        self.num_observations = 76
        self.num_actions = 23
        self.num_privileged_obs = None
        self.obs_context_len=5
        
        self.scale_base_lin_vel = 2.0
        self.scale_base_ang_vel = 0.25
        self.scale_project_gravity = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_actions = 0.25
        self.scale_base_force = 0.01
        self.scale_ref_motion_phase = 1.0

        # prepare gait commands
        self.cycle_time = 0.64
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.init_kps = np.asarray(self.configs.init_kps, dtype=np.float32)
        self.init_kds = np.asarray(self.configs.init_kds, dtype=np.float32)
        self.p_gains = np.asarray(self.configs.kps, dtype=np.float32)
        self.d_gains = np.asarray(self.configs.kds, dtype=np.float32)
        self.joint_limit_lo = np.asarray(self.configs.dof_pos_lower_limit_list, dtype=np.float32)
        self.joint_limit_hi = np.asarray(self.configs.dof_pos_upper_limit_list, dtype=np.float32)
        self.torque_limits = np.asarray(self.configs.dof_effort_limit_list, dtype=np.float32)
        
        self.soft_dof_pos_limit = 0.98
        for i in range(len(self.joint_limit_lo)):
            # soft limits
            if i != 5 and i != 11 and i !=4 and i != 10:
                m = (self.joint_limit_lo[i] + self.joint_limit_hi[i]) / 2
                r = self.joint_limit_hi[i] - self.joint_limit_lo[i]
                self.joint_limit_lo[i] = m - 0.5 * r * self.soft_dof_pos_limit
                self.joint_limit_hi[i] = m + 0.5 * r * self.soft_dof_pos_limit
        
        self.default_pre_dof_pos_np = np.asarray(self.configs.default_pre_dof_pos, dtype=np.float32)
        self.default_dof_pos_np = np.asarray(self.configs.default_dof_pos, dtype=np.float32)
        
        default_pre_dof_pos = torch.tensor(self.default_pre_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_pre_dof_pos = default_pre_dof_pos.unsqueeze(0)

        print(f"default_pre_dof_pos.shape: {self.default_pre_dof_pos.shape}")

        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)

        print(f"default_dof_pos.shape: {self.default_dof_pos.shape}")

        # prepare osbervations buffer
        self.obs_tensor = torch.zeros(1, self.num_observations*self.obs_context_len, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_buf = np.zeros(self.num_observations*self.obs_context_len, dtype=np.float32)
        self.hist_obs = np.zeros(self.num_observations*(self.obs_context_len-1), dtype=np.float32)
        self.hist_dict = {
            "actions": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
            "base_ang_vel": np.zeros(3*(self.obs_context_len-1), dtype=np.float32),
            "dof_pos": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
            "dof_vel": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
            "projected_gravity": np.zeros(3*(self.obs_context_len-1), dtype=np.float32),
            # "ref_joint_angles": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
            # "ref_joint_velocities": np.zeros(self.num_actions*(self.obs_context_len-1), dtype=np.float32),
            "ref_motion_phase": np.zeros(1*(self.obs_context_len-1), dtype=np.float32),
        }

    def init_mujoco_viewer(self):

        self.mj_model = mujoco.MjModel.from_xml_path(HUMANOID_XML)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.001
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)


        for _ in range(28):
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        self.viewer.user_scn.geoms[27].pos = [0,0,0]

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

class DeployNode():

    def __init__(self):
        super().__init__()  # type: ignore
        # init remote controller
        self.remote_controller = RemoteController() 
        # commands
        self.lin_vel_deadband = 0.1
        self.ang_vel_deadband = 0.1
        self.move_by_wireless_remote = USE_WIRELESS_REMOTE
        self.cmd_px_range = [0.1, 2.5]
        self.cmd_nx_range = [0.1, 2]
        self.cmd_py_range = [0.1, 0.5]
        self.cmd_ny_range = [0.1, 0.5]
        self.cmd_pyaw_range = [0.2, 1.0]
        self.cmd_nyaw_range = [0.2, 1.0]

        self.cmd_msg = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_() 
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)

        self.lowcmd_publisher_ = ChannelPublisher(lowcmd_topic, LowCmd)
        self.lowcmd_publisher_.Init()
        
        self.lowstate_subscriber = ChannelSubscriber(lowstate_topic, LowState)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        self.motor_pub_freq = 50
        self.dt = 1/self.motor_pub_freq

        self.wait_for_low_state()
        init_cmd_hg(self.cmd_msg, self.mode_machine_, self.mode_pr_)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # motion
        self.motion_ids = torch.arange(1).to(self.device)
        self.motion_start_times = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # config
        self.motion_config = OmegaConf.load("humanoid_sim2real/configs/g1_ref_kungfu.yaml")

        # init policy
        self.init_policy() # G1 init inside
        self.init_action_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.init_prev_action = np.zeros(self.motion_config.init_num_actions)
        self.start_policy = False
        self.init_policy_flag = False

        # for init policy
        self.init_obs = torch.zeros(1, self.motion_config.init_num_obs * self.motion_config.init_obs_context_len, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_obs_history = deque(maxlen=self.motion_config.init_obs_context_len)
        for _ in range(self.motion_config.init_obs_context_len):
            self.init_obs_history.append(torch.zeros(
                1, self.motion_config.init_num_obs, dtype=torch.float, device=self.device))
        self.init_cmd = np.array([0.0, 0, 0])

        # init motion library
        self._init_motion_lib()
        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        
        if DEBUG:
            self.env.init_mujoco_viewer()
            self.env.mj_data.qpos[7:] = np.concatenate((self.angles[:19], self.angles[22:26]), axis=0)
            self.env.mj_data.qpos[:3] = [0, 0, 0.78]
            mujoco.mj_forward(self.env.mj_model, self.env.mj_data)

            motion_res_cur = self._motion_lib.get_motion_state([0], torch.tensor([0.], device=self.device))
            ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]

            for i in range(ref_body_pos_extend.shape[0]):
            # if i in [0, 1, 4, 7, 2, 5, 8, 16, 18, 22, 17, 19, 23, 15]:  # joint for matching
                self.env.viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i].cpu() + torch.tensor([1., 0., 0.])

            tau = pd_control(np.concatenate((self.angles[:19], self.angles[22:26]), axis=0), 
                                        self.env.mj_data.qpos[7:], 
                                        np.concatenate((self.env.p_gains[:19], self.env.p_gains[22:26]), axis=0), 
                                        np.zeros(self.env.num_actions), 
                                        self.env.mj_data.qvel[6:], 
                                        np.concatenate((self.env.d_gains[:19], self.env.d_gains[22:26]), axis=0))
            self.env.mj_data.ctrl[:] = tau
                        # mj_step can be replaced with code that also evaluates
                        # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            
            self.env.viewer.sync()
            # for i, p in enumerate([self.ref_left_wrist_pos, self.ref_right_wrist_pos, self.ref_head_pos]):
            #     self.env.viewer.user_scn.geoms[i].pos = p

        # start
        self.start_time = time.monotonic()
        print("Press \"start\" to start policy")
        print("Press \"select\" for emergent stop")
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_pos_hist = []
        self.dof_vel_hist = []
        self.imu_hist = []
        self.ang_vel_hist = []
        self.foot_contact_hist = []
        self.tau_hist = []
        self.obs_hist = []

        # cmd and observation
        self.xyyaw_command = np.array([0., 0., 0.], dtype= np.float32)
        self.commands_scale = np.array([self.env.scale_base_lin_vel, self.env.scale_base_lin_vel, self.env.scale_base_ang_vel])
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((1, 3), device= self.device, dtype= torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1
        
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.phase = torch.zeros(1, device=self.device, dtype=torch.float)

        self.Emergency_stop = False
        self.stop = False

        #self.gamepad = Gamepad()
        
        time.sleep(1)

    def LowStateHgHandler(self, msg: LowState):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)
        self.joy_stick_callback(self.remote_controller)
    
    def joy_stick_callback(self, msg: RemoteController):
        if self.move_by_wireless_remote:
            # left-y for forward/backward
            ly = msg.ly
            if ly > self.lin_vel_deadband:
                vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (0, 1)
                vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
            elif ly < -self.lin_vel_deadband:
                vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband) # (-1, 0)
                vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
            else:
                vx = 0
            # left-x for side moving left/right
            lx = -msg.lx
            if lx > self.lin_vel_deadband:
                vy = (lx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
            elif lx < -self.lin_vel_deadband:
                vy = (lx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
                vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
            else:
                vy = 0
            # right-x for turning left/right
            rx = -msg.rx
            if rx > self.ang_vel_deadband:
                yaw = (rx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
            elif rx < -self.ang_vel_deadband:
                yaw = (rx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
                yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
            else:
                yaw = 0
            self.xyyaw_command = np.array([vx, vy, yaw], dtype= np.float32)
            # print(self.xyyaw_command) # TODO: remove

    def send_cmd(self, cmd: LowCmd):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.dt)
        print("Successfully connected to the robot.")

    ##############################
    # subscriber callbacks
    ##############################

    def _init_motion_lib(self):
        self.motion_config.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.motion_config["motion"], num_envs=self.env.num_envs, device=self.device)
        self._motion_lib.load_motions(random_sample=False)
            
        self.motion_res = self._motion_lib.get_motion_state(self.motion_ids, torch.tensor([0.], device=self.device))
        self.motion_len[0] = self._motion_lib.get_motion_length(self.motion_ids[torch.arange(self.env.num_envs)])
        self.motion_start_times[0] = torch.zeros(len(torch.arange(self.env.num_envs)), dtype=torch.float32, device=self.device)
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            create_zero_cmd(self.cmd_msg)
            self.send_cmd(self.cmd_msg)
            time.sleep(self.dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # moving time 3s
        total_time = 3
        num_step = int(total_time / self.dt)
    
        dof_size = HW_DOF
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(HW_DOF):
            init_dof_pos[i] = self.low_state.motor_state[i].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(HW_DOF):
                motor_idx = j
                target_pos = self.env.default_pre_dof_pos_np[j]
                self.cmd_msg.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.cmd_msg.motor_cmd[motor_idx].qd = 0
                self.cmd_msg.motor_cmd[motor_idx].kp = self.env.init_kps[j]
                self.cmd_msg.motor_cmd[motor_idx].kd = self.env.init_kds[j]
                self.cmd_msg.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.cmd_msg)
            time.sleep(self.dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button B signal for initing...")
        while self.remote_controller.button[KeyMap.B] != 1:
            if dp_node.remote_controller.button[KeyMap.select] == 1:
                self.zero_torque_state()
                print("Exit")
                exit()
            for j in range(HW_DOF):
                motor_idx = j
                self.cmd_msg.motor_cmd[motor_idx].q = self.env.default_pre_dof_pos_np[j]
                self.cmd_msg.motor_cmd[motor_idx].qd = 0
                self.cmd_msg.motor_cmd[motor_idx].kp = self.env.init_kps[j]
                self.cmd_msg.motor_cmd[motor_idx].kd = self.env.init_kds[j]
                self.cmd_msg.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.cmd_msg)
            time.sleep(self.dt)

        time.sleep(self.dt)

    def lowlevel_state_cb(self, msg: LowState):
        if self.remote_controller.button[KeyMap.B]: #if initing is pressed
            if self.init_policy_flag==False:
                print(f'Button B is pressed! Initing Policy start! Waiting for the Button start signal for start...')
            self.init_policy_flag = True

        if self.remote_controller.button[KeyMap.start]: #if start is pressed
            if self.start_policy==False:
                print(f'Policy start!')
            self.start_policy = True

        if self.remote_controller.button[KeyMap.select]: #if select is pressed
            print("Program exiting")
            self.stop = True

        # imu data
        imu_data = msg.imu_state
        self.msg_tick = msg.tick/1000
        self.roll, self.pitch, self.yaw = imu_data.rpy
        self.obs_ang_vel = np.array(imu_data.gyroscope)*self.env.scale_base_ang_vel
        self.obs_imu = np.array([self.roll, self.pitch, self.yaw])*self.env.scale_project_gravity
        
        quat_xyzw = torch.tensor([
            imu_data.quaternion[1],
            imu_data.quaternion[2],
            imu_data.quaternion[3],
            imu_data.quaternion[0],
        ], device= self.device, dtype= torch.float32).unsqueeze(0)
        self.obs_projected_gravity = quat_rotate_inverse(quat_xyzw, self.gravity_vec).squeeze(0)
    
        # termination condition
        r_threshold = abs(self.roll) > 0.6
        p_threshold = abs(self.pitch) > 0.6
        if r_threshold or p_threshold:
            self.get_logger().warning("Roll or pitch threshold reached")

        # motor data
        self.joint_tau = [msg.motor_state[i].tau_est for i in range(HW_DOF)]
        self.joint_pos = [msg.motor_state[i].q for i in range(HW_DOF)]
        self.pre_obs_joint_pos = (np.array(self.joint_pos) - self.env.default_pre_dof_pos_np) * self.env.scale_dof_pos
        self.obs_joint_pos = (np.array(self.joint_pos) - self.env.default_dof_pos_np) * self.env.scale_dof_pos
        self.joint_vel = [msg.motor_state[i].dq for i in range(HW_DOF)]
        self.obs_joint_vel = np.array(self.joint_vel) * self.env.scale_dof_vel
        self.pre_obs_joint_vel = self.obs_joint_vel.copy()

        # Joint limit check
        if self.start_policy and (((np.array(self.joint_pos)-np.array(self.env.joint_limit_lo))<0).sum() >0 or ((np.array(self.joint_pos)-np.array(self.env.joint_limit_hi))>0).sum() > 0):
            print("Joint limit reached")
            if (self.joint_pos-np.array(self.env.joint_limit_lo)<0).sum() >0:
                idx = np.where((np.array(self.joint_pos)-np.array(self.env.joint_limit_lo))<0)[0]
                print("Low limit Joint index: ", idx, self.joint_pos[idx[0]], np.array(self.env.joint_limit_lo)[idx[0]])
            if (self.joint_pos-np.array(self.env.joint_limit_hi)>0).sum() > 0:
                idx = np.where((np.array(self.joint_pos)-np.array(self.env.joint_limit_hi))>0)[0][0]
                print("High limit Joint index: ", idx, self.joint_pos[idx[0]], np.array(self.env.joint_limit_hi)[idx[0]])
            Warning("Emergency stop")
            self.Emergency_stop = True

    ##############################
    # deploy initial policy (before target action)
    ##############################
    def init_policy(self):
        print("Preparing init policy")
        faulthandler.enable()

        # prepare environment
        self.env = G1()

        # load policy
        self.initial_policy = torch.jit.load(self.motion_config["init_policy_path"])

    def pre_policy_observations(self):
        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase)
        cos_pos = torch.cos(2 * torch.pi * phase)

        obs_buf = torch.tensor(np.concatenate((sin_pos.clone().detach().cpu().numpy(), # 1
                            cos_pos.clone().detach().cpu().numpy(), # 1
                            self.xyyaw_command * self.commands_scale, # dim 3
                            # self.obs_joint_pos[:12], # dim 12
                            # self.obs_joint_pos[15:29], # dim 14
                            # self.obs_joint_vel[:12], # dim 12
                            # self.obs_joint_vel[15:29], # dim 12
                            self.pre_obs_joint_pos[:29], # dim 29
                            self.pre_obs_joint_vel[:29], # dim 29
                            self.init_prev_action, # dim 29
                            self.obs_ang_vel,  # dim 3
                            self.obs_imu,  # 3
                            np.zeros(6), # dim 6
                            ), axis=-1), dtype=torch.float, device=self.device).unsqueeze(0)
        
        obs_now = obs_buf.clone()
        self.init_obs_history.append(obs_now)
        obs_buf_all = torch.cat([self.init_obs_history[i]
                                 for i in range(self.init_obs_history.maxlen)], dim=-1)
        self.init_obs = obs_buf_all

    def get_walking_cmd_mask(self):
        walking_mask0 = np.abs(self.xyyaw_command[0]) > 0.1
        walking_mask1 = np.abs(self.xyyaw_command[1]) > 0.1
        walking_mask2 = np.abs(self.xyyaw_command[2]) > 0.2
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2

        walking_mask = walking_mask | (self.env.gait_indices.cpu() >= self.dt / self.env.cycle_time).numpy()[0]
        walking_mask |= np.logical_or(np.abs(self.obs_imu[1])>0.1, np.abs(self.obs_imu[0])>0.05)
        return walking_mask
    
    def _get_phase(self):
        return self.env.gait_indices
    
    def step_contact_targets(self):
        cycle_time = self.env.cycle_time
        standing_mask = ~self.get_walking_cmd_mask()
        self.env.gait_indices = torch.remainder(self.env.gait_indices + self.dt / cycle_time, 1.0)
        if standing_mask:
            self.env.gait_indices[:] = 0

    ##############################
    # deploy target policy
    ##############################
    def init_action_policy(self):
        print("Preparing action policy")
        faulthandler.enable()

        # prepare environment
        # self.env = G1()

        # load policy
        self.policy = load_onnx_policy(self.motion_config["action_policy_path"])
        self.angles = self.env.default_dof_pos_np
    
    def compute_observations(self):
        """ Computes observations
        """
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        self.ref_motion_phase = motion_times / self._ref_motion_length

        self.env.obs_buf[:self.env.num_actions] = self.prev_action.copy()
        self.env.obs_buf[self.env.num_actions:self.env.num_actions+3] = self.obs_ang_vel.copy()
        self.env.obs_buf[self.env.num_actions+3 : self.env.num_actions*2+3] = np.concatenate((self.obs_joint_pos[:19], self.obs_joint_pos[22:26])).copy()
        self.env.obs_buf[self.env.num_actions*2+3 : self.env.num_actions*3+3] = np.concatenate((self.obs_joint_vel[:19], self.obs_joint_vel[22:26])).copy()
        history_numpy = []
        for key in sorted(self.env.hist_dict.keys()):
            history_numpy.append(self.env.hist_dict[key])
        self.env.obs_buf[self.env.num_actions*3+3 : self.env.num_actions*3+3+self.env.num_observations*(self.env.obs_context_len-1)] = np.concatenate(history_numpy, axis=-1)
        self.env.obs_buf[self.env.num_actions*3+3+self.env.num_observations*(self.env.obs_context_len-1): self.env.num_actions*3+6+self.env.num_observations*(self.env.obs_context_len-1)] = self.obs_projected_gravity.cpu()
        self.env.obs_buf[self.env.num_actions*3+6+self.env.num_observations*(self.env.obs_context_len-1):] = self.ref_motion_phase.cpu()
        
        self.env.obs_tensor = torch.from_numpy(self.env.obs_buf).unsqueeze(0).to(self.device)
        self.env.hist_dict["actions"] = np.concatenate([self.prev_action, self.env.hist_dict["actions"][:-self.env.num_actions]])
        self.env.hist_dict["base_ang_vel"] = np.concatenate([self.obs_ang_vel, self.env.hist_dict["base_ang_vel"][:-3]])
        self.env.hist_dict["dof_pos"] = np.concatenate([self.obs_joint_pos[:19], self.obs_joint_pos[22:26], self.env.hist_dict["dof_pos"][:-self.env.num_actions]])
        self.env.hist_dict["dof_vel"] = np.concatenate([self.obs_joint_vel[:19], self.obs_joint_vel[22:26], self.env.hist_dict["dof_vel"][:-self.env.num_actions]])
        self.env.hist_dict["projected_gravity"] = np.concatenate([self.obs_projected_gravity.cpu(), self.env.hist_dict["projected_gravity"][:-3]])
        self.env.hist_dict["ref_motion_phase"] = np.concatenate([self.ref_motion_phase.cpu(), self.env.hist_dict["ref_motion_phase"][:-1]])

    @torch.no_grad()
    def run(self):
        
        loop_start_time = time.monotonic()
        # print("start main loop")
        self.lowlevel_state_cb(self.low_state)

        if self.init_policy_flag and not self.start_policy:
            self.step_contact_targets()
            self.pre_policy_observations()

            init_actions = self.initial_policy(self.init_obs.detach().reshape(1, -1))
            if torch.any(torch.isnan(init_actions)):
                print("Emergency stop due to NaN")
                self.zero_torque_state()
                self.move_to_default_pos()
                raise SystemExit
            self.init_prev_action = init_actions.clone().detach().cpu().numpy().squeeze(0)
            init_whole_body_action = init_actions.clone().detach().cpu().numpy().squeeze(0)

            init_actions_scaled = init_whole_body_action * self.env.scale_actions
            p_limits_low = (-np.array(self.env.torque_limits)) + self.env.init_kds*self.joint_vel
            p_limits_high = (np.array(self.env.torque_limits)) + self.env.init_kds*self.joint_vel
            actions_low = (p_limits_low/self.env.init_kps) - self.env.default_pre_dof_pos_np + self.joint_pos
            actions_high = (p_limits_high/self.env.init_kps) - self.env.default_pre_dof_pos_np + self.joint_pos
            target_dof_pos = np.clip(init_actions_scaled, actions_low, actions_high) + self.env.default_pre_dof_pos_np

            for i in range(HW_DOF):
                self.cmd_msg.motor_cmd[i].q = target_dof_pos[i]
                self.cmd_msg.motor_cmd[i].qd = 0
                self.cmd_msg.motor_cmd[i].tau = 0.0
                self.cmd_msg.motor_cmd[i].kp = self.env.init_kps[i]
                self.cmd_msg.motor_cmd[i].kd = (self.env.init_kds[i])

            if not NO_MOTOR and not DEBUG:
                self.send_cmd(self.cmd_msg)
                time.sleep(self.dt)
                
        if self.init_policy_flag and self.start_policy:
            if DEBUG and SIM:
                self.lowlevel_state_mujoco()

            self.compute_observations()
            self.episode_length_buf += 1

            obs_tensor = self.env.obs_tensor.reshape(1, -1).clone().detach().cpu().numpy()#.astype(np.float32)
            raw_actions = self.policy(obs_tensor) #self.policy(self.env.obs_tensor.detach().reshape(1, -1))
            if torch.any(torch.isnan(raw_actions)):
                print("Emergency stop due to NaN")
                self.zero_torque_state()
                self.move_to_default_pos()
                raise SystemExit
            self.prev_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
            whole_body_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
            
            # whole_body_action = np.pad(whole_body_action, pad_width=padding, mode='constant', constant_values=0)
            whole_body_action  = np.concatenate((whole_body_action[:19], np.zeros(3), whole_body_action[19:23], np.zeros(3)))
            # angles = whole_body_action * self.env.scale_actions + self.env.default_dof_pos_np
            # self.angles = np.clip(angles, self.env.joint_limit_lo, self.env.joint_limit_hi)
            actions_scaled = whole_body_action * self.env.scale_actions
            p_limits_low = (-np.array(self.env.torque_limits)) + self.env.d_gains*self.joint_vel
            p_limits_high = (np.array(self.env.torque_limits)) + self.env.d_gains*self.joint_vel
            actions_low = (p_limits_low/self.env.p_gains) - self.env.default_dof_pos_np + self.joint_pos
            actions_high = (p_limits_high/self.env.p_gains) - self.env.default_dof_pos_np + self.joint_pos
            self.angles = np.clip(actions_scaled, actions_low, actions_high) + self.env.default_dof_pos_np

            if LOG_DATA:
                self.action_hist.append(self.prev_action)

            for i in range(HW_DOF):
                self.cmd_msg.motor_cmd[i].q = self.angles[i]
                self.cmd_msg.motor_cmd[i].qd = 0
                # self.cmd_msg.motor_cmd[i].dq = 0.0
                self.cmd_msg.motor_cmd[i].tau = 0.0
                self.cmd_msg.motor_cmd[i].kp = self.env.p_gains[i]
                self.cmd_msg.motor_cmd[i].kd = (self.env.d_gains[i])
            # self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        
            if not NO_MOTOR and not DEBUG:
                self.send_cmd(self.cmd_msg)
                time.sleep(self.dt)
                pass
            else:
                motion_res_cur = self._motion_lib.get_motion_state([0], (self.episode_length_buf + 1) * self.dt + self.motion_start_times)
                ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]

                for i in range(ref_body_pos_extend.shape[0]):
                # if i in [0, 1, 4, 7, 2, 5, 8, 16, 18, 22, 17, 19, 23, 15]:  # joint for matching
                    self.env.viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i].cpu() + torch.tensor([1., 0., 0.])
                if not SIM:
                    self.env.mj_data.qpos[7:] = np.concatenate((self.angles[:19], self.angles[22:26]), axis=0)
                    mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
                    self.env.viewer.sync()
                else:
                    for i in range(20):
                        self.env.viewer.sync()
                        tau = pd_control(np.concatenate((self.angles[:19], self.angles[22:26]), axis=0), 
                                            self.env.mj_data.qpos[7:], 
                                            np.concatenate((self.env.p_gains[:19], self.env.p_gains[22:26]), axis=0), 
                                            np.zeros(self.env.num_actions), 
                                            self.env.mj_data.qvel[6:], 
                                            np.concatenate((self.env.d_gains[:19], self.env.d_gains[22:26]), axis=0))
                        self.env.mj_data.ctrl[:] = tau
                        # mj_step can be replaced with code that also evaluates
                        # a policy and applies a control signal before stepping the physics.
                        mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            current_time = self.episode_length_buf * self.dt + self.motion_start_times
            if current_time > self._ref_motion_length:
                breakpoint()
            
            bar_length = 50
            progress = current_time / self._ref_motion_length
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            
            # 输出不换行的进度条，并刷新输出
            sys.stdout.write(f"\rProgress: [{bar}] {int(progress * 100)}%")
            sys.stdout.flush()

        while 0.02-time.monotonic()+loop_start_time>0:  #0.012473  0.019963
            pass
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument('--task_name', action='store', type=str, help='Task name: stand, stand_w_waist, wb, squat', required=False, default='stand')
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)
    
    # rclpy.init(args=None)
    dp_node = DeployNode()
    dp_node.zero_torque_state()
    # Move to the default position
    dp_node.move_to_default_pos()
    # Enter the default position state, press the A key to continue executing
    dp_node.default_pos_state()

    print("Deploy node started")
    while True:
        try:
            dp_node.run()
            # Press the select key to exit
            if dp_node.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(dp_node.cmd_msg)
    dp_node.send_cmd(dp_node.cmd_msg)
    print("Exit")
