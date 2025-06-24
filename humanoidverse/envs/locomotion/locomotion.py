from humanoidverse.envs.base_task.base_task import BaseTask

from humanoidverse.envs.base_task import register as base_register
import copy

base_register.coreregistry

locomotion_evaluater_registry = {}
locomotion_evaluater_registry["episode_manager"] = base_register.coreregistry["episode_manager"]
locomotion_evaluater_registry["extras_manager"] = base_register.coreregistry["extras_manager"]
locomotion_evaluater_registry["terrain_manager"] = base_register.coreregistry["terrain_manager"]
locomotion_evaluater_registry["actuators_manager"] = base_register.coreregistry["actuators_manager"]
locomotion_evaluater_registry["robotdata_manager"] = base_register.coreregistry["robotdata_manager"]
locomotion_evaluater_registry["actions_manager"] = base_register.coreregistry["actions_manager"]
locomotion_evaluater_registry["observations_manager"] = base_register.coreregistry["observations_manager"]
locomotion_evaluater_registry["robotstatus_manager"] = base_register.coreregistry["robotstatus_manager"]
locomotion_evaluater_registry["terminations_manager"] = base_register.coreregistry["terminations_manager"]

from humanoidverse.envs.locomotion.term.mdp import command  # mdp
locomotion_evaluater_registry["command_manager"] = command.VelocityCommand


locomotion_trainer_registry = copy.deepcopy(locomotion_evaluater_registry)
locomotion_trainer_registry["rewards_manager"] =  base_register.coreregistry["rewards_manager"]
locomotion_trainer_registry["feet_manager"] = base_register.coreregistry["feet_manager"]
locomotion_trainer_registry["push_manager"] =  base_register.coreregistry["push_manager"]
locomotion_trainer_registry["observations_noise_currculum"] =  base_register.coreregistry["observations_noise_currculum"]
locomotion_trainer_registry["reward_limits_curriculum"] =  base_register.coreregistry["reward_limits_curriculum"]


from humanoidverse.envs.locomotion.rewards import command
locomotion_rewards_registry = copy.deepcopy(base_register.core_rewards_registry)
locomotion_rewards_registry['command_rewards'] = command.CommandRewards

class LocomotionTrainer(BaseTask):
    def __init__(self, config, device):
        super(LocomotionTrainer, self).__init__(config, device)

    @property
    def manager_map(self):
        return locomotion_trainer_registry

    @property
    def rewards_map(self):
        return locomotion_rewards_registry


class GaitTrainer(LocomotionTrainer):
    def __init__(self, config, device):
        super(GaitTrainer, self).__init__(config, device)

    @property
    def rewards_map(self):
        _maps = super(GaitTrainer, self).rewards_map
        _maps = copy.deepcopy(_maps)
        from humanoidverse.envs.locomotion.rewards import feet_gait
        _maps['gait_rewards'] = feet_gait.GaitRewards
        return _maps

class LocomotionEvaluater(BaseTask):
    def __init__(self, config, device):
        super(LocomotionEvaluater, self).__init__(config, device)
        self.debug_viz = True

    @property
    def manager_map(self):
        return locomotion_evaluater_registry

    @property
    def rewards_map(self):
        return {}

class GaitEvaluater(LocomotionEvaluater):
    def __init__(self, config, device):
        super(GaitEvaluater, self).__init__(config, device)
