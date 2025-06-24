import torch
from humanoidverse.envs.base_task.term import base

class FeetManager(base.BaseManager):
    def __init__(self, _task):
        super(FeetManager, self).__init__(_task)

    # stage 1
    def post_init(self):
        # wait for robotdata_manager init
        robotdata_manager = self.task.robotdata_manager

        self.contacts = torch.zeros(self.num_envs, len(robotdata_manager.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contacts_filt = torch.zeros(self.num_envs, len(robotdata_manager.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)


        self.feet_air_max_height = torch.zeros(self.num_envs, robotdata_manager.feet_indices.shape[0],
                                               dtype=torch.float, device=self.device, requires_grad=False)

        ##
        self.feet_air_time = torch.zeros(self.num_envs, robotdata_manager.feet_indices.shape[0],
                                         dtype=torch.float, device=self.device, requires_grad=False)

    # stage 3
    def pre_compute(self):
        robotdata_manager = self.task.robotdata_manager
        ## contact update
        contact = self.task.simulator.contact_forces[:, robotdata_manager.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.contacts)
        self.contacts = contact
        self.contact_filt = contact_filt

        ## max_height
        self.feet_air_max_height = torch.max(self.feet_air_max_height, \
                                             self.task.simulator._rigid_body_pos[:, robotdata_manager.feet_indices, 2])

        self.feet_air_time += self.task.dt  ## TODO update here


    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.feet_air_time[env_ids] = 0.

    def post_step(self):
        self.feet_air_max_height *= ~self.contact_filt
        self.last_contacts_filt = self.contact_filt
        self.feet_air_time *= ~self.contact_filt

    ######################### Observations #########################
    def _get_obs_feet_contact_force(self,):
        robotdata_manager = self.task.robotdata_manager
        return self.simulator.contact_forces[:, robotdata_manager.feet_indices, :].view(self.num_envs, -1)

