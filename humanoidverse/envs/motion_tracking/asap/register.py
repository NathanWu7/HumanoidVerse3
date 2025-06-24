from humanoidverse.envs.base_task import register as base_register
import copy

base_register.coreregistry

from humanoidverse.envs.motion_tracking.asap.status import asap_robotstatus
from humanoidverse.envs.motion_tracking.asap.foundation import asap_robotdata
from humanoidverse.envs.motion_tracking.asap.termination import asap_termination

## evaluater
asap_evaluater_registry = {}
asap_evaluater_registry["episode_manager"] = base_register.coreregistry["episode_manager"]
asap_evaluater_registry["extras_manager"] = base_register.coreregistry["extras_manager"]
asap_evaluater_registry["terrain_manager"] = base_register.coreregistry["terrain_manager"]
asap_evaluater_registry["actuators_manager"] = base_register.coreregistry["actuators_manager"]
asap_evaluater_registry["robotdata_manager"] = asap_robotdata.AsapRobotDataEvaluater  #base_register.coreregistry["robotdata_manager"]
asap_evaluater_registry["actions_manager"] = base_register.coreregistry["actions_manager"]
asap_evaluater_registry["observations_manager"] = base_register.coreregistry["observations_manager"]
asap_evaluater_registry["robotstatus_manager"] = asap_robotstatus.AsapStatus  #base_register.coreregistry["robotstatus_manager"]
asap_evaluater_registry["terminations_manager"] = base_register.coreregistry["terminations_manager"]

## asap_termination
asap_evaluater_registry["asap_termination"] = asap_termination.TrackTermination
from humanoidverse.envs.motion_tracking.asap.extends import asap_motion_save
asap_evaluater_registry["asap_motion_save"] = asap_motion_save.MotionSave


## trainer
asap_trainer_registry = copy.deepcopy(asap_evaluater_registry)
asap_trainer_registry["robotdata_manager"] = asap_robotdata.AsapRobotData
asap_trainer_registry["rewards_manager"] =  base_register.coreregistry["rewards_manager"]
asap_trainer_registry["feet_manager"] = base_register.coreregistry["feet_manager"]
asap_trainer_registry["push_manager"] =  base_register.coreregistry["push_manager"]
asap_trainer_registry["observations_noise_currculum"] =  base_register.coreregistry["observations_noise_currculum"]
asap_trainer_registry["reward_limits_curriculum"] =  base_register.coreregistry["reward_limits_curriculum"]

from humanoidverse.envs.motion_tracking.asap.curriculum import terminate_when_motion_far_curriculum
asap_trainer_registry["terminate_when_motion_far_curriculum"] = terminate_when_motion_far_curriculum.MotionfarCurrculum

# rewards
from humanoidverse.envs.motion_tracking.asap.rewards import asap_motionstatus
asap_rewards_registry = copy.deepcopy(base_register.core_rewards_registry)
asap_rewards_registry['asap_motionstatus_rewards'] = asap_motionstatus.ASAPStatusRewards


## player
asap_player_registry = {}
asap_evaluater_registry["terrain_manager"] = base_register.coreregistry["terrain_manager"]
asap_evaluater_registry["actuators_manager"] = base_register.coreregistry["actuators_manager"]
asap_evaluater_registry["robotdata_manager"] = asap_robotdata.AsapRobotDataPlayer  #base_register.coreregistry["robotdata_manager"]
############ REWARDS ############

