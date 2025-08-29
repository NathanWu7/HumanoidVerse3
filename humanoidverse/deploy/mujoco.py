# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

import mujoco
import mujoco.viewer
import glfw
import os
import time
import signal
import sys
from pathlib import Path

import torch
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
import logging
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict, Optional, Tuple
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
from humanoidverse.utils.helpers import np2torch, torch2np
from humanoidverse.utils.real.rotation_helper import *
from humanoidverse.utils.noise_tool import noise_process_dict, RadialPerturbation

class RobotExitException(Exception):
    pass


import numpy as np

def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles



class MViewerPlugin:
    # TODO: visualize the motion keypoint in MujocoViewer
    is_pause = False
    
    def _make_viewer(self):
        def _key_callback(key):
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------
                key_char = chr(key)
                
                print(key, key_char)
                
                if key == 27: # Esc
                    self.viewer.close()
                
                if key_char == ' ':
                    self.is_pause = not self.is_pause
                
                if key_char == 'w':
                    self.cmd[1] += 0.1
                elif key_char == 's':
                    self.cmd[1] -= 0.1
                elif key_char == 'a': #
                    self.cmd[0] += 0.1
                elif key_char == 'd':
                    self.cmd[0] -= 0.1
                elif key_char == 'e':
                    if self.heading_cmd:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/20)
                    else:
                        self.cmd[2] += 0.1
                elif key_char == 'q': 
                    if self.heading_cmd:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/20)
                    else:
                        self.cmd[2] -= 0.1
                elif key_char == 'r':
                    self.cmd = np.array(self.cfg.deploy.defcmd)
                elif key == 257: # Enter
                    # self.Reset()
                    self._ref_pid = -2
                elif key_char == '[':
                    self._ref_pid -= 1
                elif key_char == ']':
                    self._ref_pid += 1
                
                print(self.cmd, self._ref_pid)
        # glfw.set_key_callback(self.viewer.window, _key_callback)
        ...
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=_key_callback)
        self.viewer.cam.lookat[:] = np.array([0,0,0.8])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 30                         # 可根据需要调整角度
        self.viewer.cam.elevation = -30                      # 负值表示从上往下看
        
        
    def render_step(self):
        while self.is_pause:
            time.sleep(0.01)
        
        if self.is_render:
            if self.viewer.is_running():
                self.viewer.sync()
            
                # breakpoint()
            else:
                raise RobotExitException("Mujoco Robot Exit")
        ...


class MujocoRobot(URCIRobot, MViewerPlugin):
    REAL=False
    
    HANG=False
    
    RAND_NOISE  : bool = False
    RAND_DELAY  : bool = True
    RAND_MASK   : bool = False
    RAND_IMU    : bool = False
    RAND_RADIAL : bool = False
    RAND_OFFSET : bool = False
    TANH_ATTACK : bool = False
    
    noise_imu_type:str = 'ou' # 'white' or 'ou' or 'pink'
    noise_imu_rpy:float = 3         # degree
    noise_imu_omega:float = 0.2     # rad/s
    # noise_imu_omega:float = 0.3     # rad/s
    noise_imu_ou_theta:float = 1    # 1/s
    # noise_imu_ou_theta:float = 0.3    # 1/s
    
    noise_ratio = 3e-2
    # delay_ratio = (4, 25) # unit: ms
    delay_ratio = (4, 20) # unit: ms
    mask_ratio = 0.7
    mk_rand_noise = lambda tens, ratio: (
                                            (tens * (1 + torch.randn_like(tens) * ratio) ) 
                                    if isinstance(tens, torch.Tensor) 
                                    else    (tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio)) )# type: ignore
    # mk_rand_noise = lambda tens, ratio: tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio) # type: ignore
    
    # print_torque = lambda tau: print(f"tau (norm, max) = {torch.norm(tau):.2f}, \t{torch.max(tau):.2f}", end='\r')
    print_torque = lambda tau: print(f"tau (norm, max) = {np.linalg.norm(tau):.2f}, \t{np.max(tau):.2f}", end='\r')
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        def signal_handler(sig, frame):
            logger.info("Ctrl+C  Exiting safely...")
            raise RobotExitException("Mujoco Robot Exiting")
        signal.signal(signal.SIGINT, signal_handler)
        
        self.decimation = cfg.simulator.config.sim.control_decimation
        self.sim_dt = 1/cfg.simulator.config.sim.fps
        assert self.dt == self.decimation * self.sim_dt
        # self._subtimer = 0
        
        
        self.model = mujoco.MjModel.from_xml_path(os.path.join(cfg.robot.asset.asset_root, cfg.robot.asset.xml_file)) # type: ignore
        print("XML", cfg.robot.asset.xml_file)
        self.data = mujoco.MjData(self.model) # type: ignore
        self.model.opt.timestep = self.sim_dt
        print("timestep", self.model.opt.timestep)
        if cfg.deploy.render:
            self.is_render = True
            if cfg.__contains__('recording') and cfg.recording:
                self.is_recording = True
            self._make_viewer()
        else:
            self.is_render = False
            
        self.num_ctrl = self.data.ctrl.shape[0]
        assert self.num_ctrl == self.num_actions, f"Number of control DOFs {self.num_ctrl} does not match number of actions {self.num_actions}"
        
        # noise init
        if self.RAND_IMU:
            if self.noise_imu_type == 'ou':
                sigma = np.sqrt(2*self.noise_imu_ou_theta)
                theta = self.noise_imu_ou_theta
            elif self.noise_imu_type == 'white':
                sigma = 1
                theta = 0
            else:
                raise ValueError(f"Invalid noise type: {self.noise_imu_type}")
            self.noise_imu = noise_process_dict[self.noise_imu_type]((6,), mu=0,sigma=sigma, theta=theta, dt=self.sim_dt)
            
            # self.noise_imu = noise_process_dict['empty']((3,))
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=250*np.sqrt(0.02), theta=25, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=250, theta=25, dt=self.sim_dt)
            
            # self.noise_imu = noise_process_dict['ou']((6,), mu=0, sigma=1, theta=0.15, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((6,), mu=0, sigma=1.825)
            
            # self.noise_imu = noise_process_dict['ou']((6,), mu=0, sigma=np.sqrt(2*3), theta=3, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((6,), mu=0, sigma=1)
            
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=np.sqrt(250), theta=25, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=35.355)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=5)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=2.236)
            print(self.noise_imu)
            # print('OUProcess', 'mu', self.noise_imu.mu, 'sigma', self.noise_imu.sigma, 'theta', self.noise_imu.theta, 'dt', self.noise_imu.dt)
            # self.noise_imu = noise_process_dict[self.noise_imu_type]((2,), mu=0, sigma=1.0, theta=self.noise_imu_omega, dt=self.sim_dt)
        
        if self.RAND_RADIAL:
            self._radial_perturbation = RadialPerturbation(sigma=0.03, kappa=100)
            self.radial_perturbation = lambda x: self._radial_perturbation(torch.from_numpy(x).reshape(1, -1)).numpy().reshape(-1)
        
        logger.info("Initializing Mujoco Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        
        logger.info(f"decimation: {self.decimation}, sim_dt: {self.sim_dt}, dt: {self.dt}")
        logger.info(f"xml_file: {cfg.robot.asset.xml_file}")
        # print(self.decimation, self.sim_dt, self.dt)
        self.Reset()
        
        mujoco.mj_step(self.model, self.data) # type: ignore    


    def _reset(self):
        self.data.qpos[:3] = np.array(self.cfg.robot.init_state.pos)
        self.data.qpos[3:7] = np.array(self.cfg.robot.init_state.rot)[[3,0,1,2]] # XYZW to WXYZ
        # self.data.qpos[3:7] = np.array([0,0,0.7184,-0.6956])[[3,0,1,2]]  #DEBUG: init quat of JingjiTaiji
        # self.data.qpos[3:7] = np.array([0,0,0.7455,-0.6665])[[3,0,1,2]]  #DEBUG: init quat of NewTaiji
        # self.data.qpos[3:7] = np.array([0,0,0.6894,0.7244])[[3,0,1,2]]  #DEBUG: init quat of Shaolinquan
        self.data.qpos[7:] = self.dof_init_pose
        self.data.qvel[:]   = 0
        self.cmd = np.array(self.cfg.deploy.defcmd)
        
        if self.RAND_IMU:
            self.noise_imu.reset()


    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        if MujocoRobot.RAND_NOISE:
            kp,kd = MujocoRobot.mk_rand_noise(np.array([kp, kd]), MujocoRobot.noise_ratio)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def _get_state(self):
        '''Extracts physical states from the mujoco data structure
        '''
        data = self.data
        self.q_raw = data.qpos.astype(np.double)[7:] # 19 dim
        self.q = self.q_raw.copy()
            # WXYZ
            # 3 dim base pos + 4 dim quat + 12 dim actuation angles
        self.dq_raw = data.qvel.astype(np.double)[6:] # 18 dim ?????
        self.dq = self.dq_raw.copy()
            # 3 dim base vel + 3 dim omega + 12 dim actuation vel
        
        self.pos = data.qpos.astype(np.double)[:3]
        self.quat_raw = data.qpos.astype(np.double)[3:7][[1,2,3,0]] # WXYZ to XYZW
        self.quat = self.quat_raw.copy()
        self.vel = data.qvel.astype(np.double)[:3]
        self.omega_raw = data.qvel.astype(np.double)[3:6]
        self.omega = self.omega_raw.copy()
        
        # print(self.omega_raw)
        if self.RAND_OFFSET:
            self.q = self.q_raw - self._motor_offset
        
        if self.RAND_IMU:
            step_noise_imu = self.noise_imu.x
            
            self.rpy = (quaternion_to_euler_array(self.quat)) + step_noise_imu[:3] * self.noise_imu_rpy *(np.pi/180)  
            self.quat = rpy_to_quaternion_array(self.rpy)
            quat_dist = np.arccos(np.clip(np.abs(np.dot(self.quat_raw, self.quat)), -1.0, 1.0))
            
            self.omega = self.omega + step_noise_imu[3:] * self.noise_imu_omega
            
            # print('quat_dist:', quat_dist)
            # print('rpy', self.rpy)
            # print('omega', self.omega, '\t| omega_raw', self.omega_raw, '\t| step_noise_imu', step_noise_imu[3:])
            # breakpoint()
            
        if self.RAND_RADIAL:
            # self.omega = self.radial_perturbation(self.omega)
            self.dq = self.radial_perturbation(self.dq)
            
            # print(f"{self.omega=}, {self.omega_raw=}")
            print(f"{self.dq=}, {self.dq_raw=}")
        
        r = R.from_quat(self.quat)  # R.from_quat: need xyzw
        self.rpy = quaternion_to_euler_array(self.quat) # need xyzw
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        self.gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
        
    
    def _sanity_check(self, target_q):
        unsafe_dof = np.where((np.abs(target_q - self.q) > 2.2 ) | 
                              (np.abs(self.dq) > 20))[0]
        if len(unsafe_dof) > 0:
            for motor_idx in unsafe_dof:
                logger.error(f"Action of joint {motor_idx} is too large.\n"
                                f"target q\t: {target_q[motor_idx]} \n"
                                f"target dq\t: {0} \n"
                                f"q\t\t: {self.q[motor_idx]} \n"
                                f"dq\t\t: {self.dq[motor_idx]}\n")
                # breakpoint()  
        
    _motor_offset = np.array([
        3, 0.5, 2, -0.5, -1, 1,
        -2, 1, -.3, 1, 0.3, 0.1,
        0, 0, 0,
        0, 1, 0, -1,
        -2, 0, 0, 0
    ])*(np.pi/180)   # [23]
        
    def _apply_action(self, target_q):
        
        self._sanity_check(target_q)
        
        rand_mask = np.random.random(self.num_actions) < self.mask_ratio

        rand_delay = np.random.randint((1e-3*self.delay_ratio[0])//self.sim_dt, (1e-3*self.delay_ratio[1])//self.sim_dt) * self.RAND_DELAY
        step_delay = rand_delay//self.decimation
        substep_delay = rand_delay - step_delay * self.decimation
        
        if step_delay ==0:
            old_action = self.history_handler.history['actions'][0, step_delay]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = target_q
        else:
            old_action = self.history_handler.history['actions'][0, step_delay]
            cur_action = self.history_handler.history['actions'][0, step_delay+1]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = np.clip(cur_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
                
        for i in range(self.decimation):
            self.GetState()
            
            if self.RAND_DELAY and i < substep_delay:
                target_q = old_trg_q
            elif self.RAND_MASK:
                target_q = cur_trg_q * rand_mask + old_trg_q.numpy() * (1 - rand_mask)
            else:
                target_q = cur_trg_q
                
            # if self.RAND_OFFSET:
            #     target_q = target_q + self._motor_offset
                
            tau = self.pd_control(target_q, self.q, self.kp,
                            0, self.dq, self.kd)  # Calc torques
            
            if self.TANH_ATTACK:
                kappa = {
                    4:15,
                    10:15,
                    13:200,
                    # 14:200,
                }
                # tau = np.tanh(tau)
                for i in kappa.keys():
                    tau[i] = np.tanh(tau[i]/kappa[i]) * kappa[i]
            
            if self.RAND_NOISE: tau = MujocoRobot.mk_rand_noise(tau, MujocoRobot.noise_ratio)
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clamp torques
            
            
            # MujocoRobot.print_torque(tau)
            # tau*=0
            # print(np.linalg.norm(target_q-self.q), np.linalg.norm(self.dq), np.linalg.norm(tau))
            # self.data.qpos[:3] = np.array([0,0,1])
            self.data.ctrl[:] = tau
            if self.HANG:
                # self.data.ctrl[14] = 0.5
                self.data.qpos[:3] = np.array([0,0,1])
                self.data.qpos[3:7] = np.array([1,0,0,0])

            mujoco.mj_step(self.model, self.data) # type: ignore
    
            if self.RAND_IMU:
                self.noise_imu.step()
            
            self.tracking()
            self.render_step()
            # self._subtimer += 1
        
        
    # Noise Version ApplyAction & Obs
    def ApplyAction(self, action): 
        if self.RAND_NOISE: action = MujocoRobot.mk_rand_noise(action, MujocoRobot.noise_ratio)
        URCIRobot.ApplyAction(self, action)


            
        # breakpoint()

    def Obs(self):
        
        # return {k: torch2np(v) for k, v in self.obs_buf_dict.items()}
        
        actor_obs = torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)
        if self.RAND_NOISE:
            actor_obs = MujocoRobot.mk_rand_noise(actor_obs, MujocoRobot.noise_ratio)
        return {
            'actor_obs': actor_obs
            }
        

    def _get_motion_to_save_torch(self)->Tuple[float, Dict[str, torch.Tensor]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()).float() # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = torch.cat([root_rot_vec[..., None, :], 
                                torch.from_numpy(self._dof_axis * dof[..., None]), 
                                torch.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': torch.from_numpy(root_trans),
            'root_rot': torch.from_numpy(root_rot), # 统一save xyzw
            'dof': torch.from_numpy(dof),
            'pose_aa': pose_aa,
            'action': torch.from_numpy(self.act),
            'actor_obs': np2torch(self.obs_buf_dict['actor_obs']),
            'terminate': torch.zeros((1,)),
            'dof_vel': torch.from_numpy(self.dq),
            'root_lin_vel': torch.from_numpy(self.vel),
            'root_ang_vel': torch.from_numpy(self.omega),
            
            'clock_time': torch.from_numpy(np.array([time.time()])),
            'tau': torch.from_numpy(self.data.ctrl).clone(),
            'cmd': torch.from_numpy(self.cmd),
            'root_rot_raw': torch.from_numpy(self.quat_raw),
            'root_ang_vel_raw': torch.from_numpy(self.omega_raw),
        }
        
    def _get_motion_to_save_np(self)->Tuple[float, Dict[str, np.ndarray]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = np.array(sRot.from_quat(root_rot).as_rotvec(), dtype=np.float32) # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = np.concatenate([root_rot_vec[..., None, :], 
                                np.array(self._dof_axis * dof[..., None]), 
                                np.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': (root_trans).copy(),
            'root_rot': (root_rot).copy(), # 统一save xyzw
            'dof': (dof).copy(),
            'pose_aa': pose_aa,
            'action': (self.act).copy(),
            'actor_obs': (self.obs_buf_dict['actor_obs']),
            'terminate': np.zeros((1,)),
            'dof_vel': (self.dq).copy(),
            'root_lin_vel': (self.vel).copy(),
            'root_ang_vel': (self.omega).copy(),
            
            'clock_time': (np.array([time.time()])),
            'tau': (self.data.ctrl).copy(),
            'cmd': (self.cmd).copy()
        }
        
    _get_motion_to_save = _get_motion_to_save_np


    def tracking(self):
        # breakpoint()
        # print(np.linalg.norm(self.data.xpos[6]-self.data.xpos[12]))
        if np.any(self.data.contact.pos[:,2] > 0.01):
            names_list = self.model.names.decode('utf-8').split('\x00')[:40]
            res = np.zeros((6,1),dtype=np.float64)
            geom_name = lambda x: (names_list[self.model.geom_bodyid[x] + 1])
            geom_force = lambda x:mujoco.mj_contactForce(self.model,self.data,x,res) #type:ignore
            
            for contact in self.data.contact:
                if contact.pos[2] > 0.01 and contact.geom1 != 0 and contact.geom2 != 0:
                    geom1_name = geom_name(contact.geom1)
                    geom2_name = geom_name(contact.geom2)
                    logger.warning(f"Warning!!! Collision between '{geom1_name,contact.geom1}' and '{geom2_name,contact.geom2}' at position {contact.pos}.")
                    # breakpoint()
                    
