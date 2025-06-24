from humanoidverse.agents.base_algo import register
from humanoidverse.agents.ppo.logic import rollout, trainer, statistics, evaluater
from humanoidverse.agents.ppo.component import envwarp, modules, optimizer, storage

ppo_trainer_registry: dict = {}
ppo_evaluater_registry: dict = {}

################################################
ppo_trainer_registry["rollout_component"] = rollout.Rollout
ppo_trainer_registry["trainer_component"] = trainer.Trainer
ppo_trainer_registry["statistics_component"] = statistics.Statistics

ppo_evaluater_registry["evaluater_component"] = evaluater.Evaluater
################################################
ppo_trainer_registry["envwarp_component"] = envwarp.EnvWarp
ppo_trainer_registry["optimizer_component"] = optimizer.Optimizer
ppo_trainer_registry["modules_component"] = modules.Modules # moduler
ppo_trainer_registry["storage_component"] = storage.Storage

ppo_evaluater_registry["envwarp_component"] = envwarp.EnvWarp
ppo_evaluater_registry["modules_component"] = modules.Modules # moduler

trainer_namespace: str  = "ppo_trainer"
evaluater_namespace: str  = "ppo_evaluater"
################################################
register.registry[trainer_namespace] = ppo_trainer_registry
register.registry[evaluater_namespace] = ppo_evaluater_registry
