from humanoidverse.envs.base_task.term.statistics import extras
from humanoidverse.envs.base_task.term.foundation import episode, robotdata, terrain, actuators
from humanoidverse.envs.base_task.term.mdp import actions, observations, rewards
from humanoidverse.envs.base_task.terminates import terminations
from humanoidverse.envs.base_task.term.sim2real import push

coreregistry: dict = {} # [str, base.BaseManager] = {}

coreregistry["episode_manager"] = episode.BaseEpisode                 # core
coreregistry["terrain_manager"] = terrain.BaseTerrainManager          # core
coreregistry["actuators_manager"] = actuators.ActuatorsManager        # core
coreregistry["robotdata_manager"] = robotdata.RandResetDataManager    # core

coreregistry["actions_manager"] = actions.ActionsManager
coreregistry["observations_manager"] = observations.BaseObservations

from humanoidverse.envs.base_task.term.status import feet, robotstatus # status
coreregistry["robotstatus_manager"] = robotstatus.StatusManager
coreregistry["feet_manager"] = feet.FeetManager

coreregistry["rewards_manager"] = rewards.BaseRewardsManager
coreregistry["extras_manager"] = extras.BaseExtrasManager


coreregistry["push_manager"] = push.PushManager
coreregistry["terminations_manager"] = terminations.TerminateManager

from humanoidverse.envs.base_task.currculum import observations_noise_currculum, reward_limits_curriculum
coreregistry["observations_noise_currculum"] = observations_noise_currculum.NoiseCurrculum
coreregistry["reward_limits_curriculum"] = reward_limits_curriculum.LimitsCurrculum

############ REWARDS ############
from humanoidverse.envs.base_task.rewards import actions, episode, feet, robotstatus, actuators, body
core_rewards_registry: dict = {}
core_rewards_registry['actions_rewards'] = actions.ActionsRewards
core_rewards_registry['episode_rewards'] = episode.EpisodeRewards
core_rewards_registry['body_rewards'] = body.BodyRewards
core_rewards_registry['feet_rewards'] = feet.FeetRewards
core_rewards_registry['robotstatus_rewards'] = robotstatus.StatusRewards
core_rewards_registry['actuators_rewards'] = actuators.ActuatorsRewards

