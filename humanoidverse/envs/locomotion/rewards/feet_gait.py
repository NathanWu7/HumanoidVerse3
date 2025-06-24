from humanoidverse.envs.base_task.term import base
import torch
import numpy as np
from scipy.stats import vonmises

class GaitRewards(base.BaseManager):

    def __init__(self, _task):
        super(GaitRewards, self).__init__(_task)

    # stage 1
    def init(self):
        self.a_swing = 0.0 # start of the swing phase
        self.b_swing = 0.5 # end of the swing phase
        self.a_stance = 0.5 # start of the stance phase
        self.b_stance = 1.0 # end of the stance phase
        self.kappa = 4.0 # shared variance in Von Mises
        self.left_offset = 0.0 # left foot offset
        self.right_offset = 0.5 # right foot offset

        self.left_feet_height = torch.zeros(self.num_envs, device=self.device) # left feet height
        self.right_feet_height = torch.zeros(self.num_envs, device=self.device) # right feet height

        self.phase_time = torch.zeros(self.num_envs, dtype=torch.float32, requires_grad=False, device=self.device)
        self.phase_time_np = np.zeros(self.num_envs, dtype=np.float32)
        self.phase_left = (self.phase_time + self.left_offset) % 1
        self.phase_right = (self.phase_time + self.right_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        # Initialize the gait period
        if hasattr(self.config.rewards, "gait_period"):
            if not self.config.rewards.gait_period:
                self.T = self.config.rewards.gait_period # gait period in seconds # gait period in seconds
            else:
                self.T = 1. # gait period in seconds
        else:
            self.T = 1.

        if hasattr(self.config.rewards, "gait_period"):
            # Randomize the gait phase time
            if self.config.obs.use_phase:
                self.phi_offset = np.random.rand(self.num_envs)*self.T
            else:
                self.phi_offset = np.zeros(self.num_envs)
        else:
            self.phi_offset = np.zeros(self.num_envs)
        # Initialize the target arm joint positions
        self.swing_arm_joint_pos = torch.tensor([-1.04, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        self.stance_arm_joint_pos = torch.tensor([0.757, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        print("phi_offset: ", self.phi_offset)

    # stage 3
    def pre_compute(self):
        self.phase_time_np = self._calc_phase_time()
        self.phase_time = torch.tensor(self.phase_time_np, device=self.device, dtype=torch.float, requires_grad=False)
        self.phase_left = (self.phase_time + self.left_offset) % 1
        self.phase_right = (self.phase_time + self.right_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

    ########################### GAIT REWARDS ###########################
    def _calc_phase_time(self):
        # Calculate the phase time
        episode_manager = self.task.episode_manager

        episode_length_np = episode_manager.episode_length_buf.cpu().numpy()
        phase_time = (episode_length_np * self.task.dt + self.phi_offset) % self.T / self.T
        return phase_time

    def _reward_gait_contact(self):
        robotdata_manager = self.task.robotdata_manager

        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2): # left and right feet
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.task.simulator.contact_forces[:, robotdata_manager.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def calculate_phase_expectation(self, phi, offset=0, phase="swing"):
        """
        Calculate the expectation value of I_i(φ).

        Parameters:
        phi (float): The given phase time.
        offset (float): The offset of the phase time.

        Returns:
        float: The expectation value of I_i(φ).
        """
        # print("phase_time: ", phi)
        phi = (phi + offset) % 1
        phi *= 2 * np.pi
        # Create Von Mises distribution objects for A_i and B_i
        if phase == "swing":
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_swing)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_swing)
        else:
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_stance)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_stance)
        # Calculate P(A_i < φ) and P(B_i < φ)
        P_A_less_phi = dist_A.cdf(phi)
        P_B_less_phi = dist_B.cdf(phi)
        # Calculate P(A_i < φ < B_i)
        P_A_phi_B = P_A_less_phi * (1 - P_B_less_phi)
        # Calculate the expectation value of I_i
        E_I_i = P_A_phi_B

        return E_I_i

    def _reward_gait_period(self):
        """
        Jonah Siekmann, et al. "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
        paper link: https://arxiv.org/abs/2011.01387
        """
        robotdata_manager = self.task.robotdata_manager

        # Calculate the expectation value of I_i of left and right feet
        E_I_l_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="swing")
        E_I_l_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="stance")
        E_I_r_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="swing")
        E_I_r_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="stance")
        # print("E_I_l_swing: ", E_I_l_swing, ", E_I_r_swing: ", E_I_r_swing)
        # print("E_I_l_stance: ", E_I_l_stance, ", E_I_r_stance: ", E_I_r_stance)
        ## Convert to tensor
        E_I_l_swing = torch.tensor(E_I_l_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_swing = torch.tensor(E_I_r_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_l_stance = torch.tensor(E_I_l_stance, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_stance = torch.tensor(E_I_r_stance, device=self.device, dtype=torch.float, requires_grad=False)
        # Get the contact forces and velocities of the feet, and the velocities of the arm ee
        Ff_left = torch.norm(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices[0], :], dim=-1) # left foot contact force
        Ff_right = torch.norm(self.task.simulator.contact_forces[:, robotdata_manager.feet_indices[1], :], dim=-1) # right foot contact force
        vf_left = torch.norm(self.task.simulator._rigid_body_vel[:, robotdata_manager.feet_indices[0], :], dim=-1) # left foot velocity
        vf_right = torch.norm(self.task.simulator._rigid_body_vel[:, robotdata_manager.feet_indices[1], :], dim=-1) # right foot velocity
        # print("Ff_left: ", Ff_left, ", Ff_right: ", Ff_right)
        # print("vf_left: ", vf_left, ", vf_right: ", vf_right)
        reward_gait = E_I_l_swing * torch.exp(-Ff_left**2) + E_I_r_swing * torch.exp(-Ff_right**2) + \
                      E_I_l_stance * torch.exp(-200*vf_left**2) + E_I_r_stance * torch.exp(-200*vf_right**2)
        # Sum up the gait reward
        return reward_gait

