import torch
from humanoidverse.agents.base_algo import base, base_algo

from humanoidverse.envs.base_task.base_task import BaseTask
import time
import os
from loguru import logger

class BasePPO(base_algo.BaseAlgo):

    def __init__(self, env: BaseTask, config, log_dir=None, device='cpu'):

        self.log_dir = log_dir
        self.current_learning_iteration = 0

        self.save_interval = config.save_interval
        # Training related Config
        self.num_steps_per_env = config.num_steps_per_env
        self.num_learning_iterations = config.num_learning_iterations

        super(BasePPO, self).__init__(env, config, device)

    def setup(self):
        # import ipdb; ipdb.set_trace()
        logger.info("Setting up PPO")
        self._pre_init()
        self._init()
        self._post_init()

    def _pre_init(self):
        for _key in self.components:
            _component = self.components[_key]
            _component.pre_init()

    def _init(self):
        for _key in self.components:
            _component = self.components[_key]
            _component.init()

    def _post_init(self):
        for _key in self.components:
            _component = self.components[_key]
            _component.post_init()

    def load(self, ckpt_path):
        # import ipdb; ipdb.set_trace()
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)

            for _key in self.components:
                _component = self.components[_key]
                _component.load(loaded_dict)

            self.current_learning_iteration = loaded_dict["iter"]

    def save(self, path):
        logger.info(f"Saving checkpoint to {path}")
        _dicts = {'iter': self.current_learning_iteration}

        for _key in self.components:
            _component = self.components[_key]
            _component.update(_dicts)

        torch.save(_dicts, path)

    def learn(self):
        assert hasattr(self, "envwarp_component")
        assert hasattr(self, "modules_component")
        assert hasattr(self, "statistics_component")

        ##
        # envwarp_component
        # modules_component suport pre_loop
        _inputs = {'context': base.Context.TRAIN.value }
        for _key in self.components:
            _component = self.components[_key]
            _component.pre_loop(_inputs)

        ## init status
        num_learning_iterations = self.num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations

        # do not use track, because it will confict with motion loading bar
        # for it in track(range(self.current_learning_iteration, tot_iter), description="Learning Iterations"):
        for it in range(self.current_learning_iteration, tot_iter):

            self.statistics_component.pre_epoch()

            _start_time = time.time()

            # Jiawei: Need to return obs_dict to update the obs_dict for the next iteration
            # Otherwise, we will keep using the initial obs_dict for the whole training process
            self._rollout_epoch()

            _time = time.time()
            _collection_time = _time - _start_time
            _start_time = _time

            self._training_epoch()

            _time = time.time()
            _learn_time = _time - _start_time

            # Logging
            log_dict = {
                'it': it,
                'collection_time': _collection_time,
                'learn_time': _learn_time,
                'num_learning_iterations': num_learning_iterations
            }

            self.statistics_component.post_epoch(log_dict)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    @torch.inference_mode()
    def _rollout_epoch(self):
        ## TODO
        self.rollout_component.pre_epoch()
        for i in range(self.num_steps_per_env):
            self.rollout_component.step()
        # prepare data for training
        self.rollout_component.post_epoch()

    def _training_epoch(self):
        # pre loop
        self.trainer_component.pre_epoch()

        _inputs = {'context': base.Context.TRAIN.value }
        self.storage_component.pre_epoch(_inputs)
        generator = _inputs['generator']

        for policy_state_dict in generator:
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)

            _inputs = {'context': base.Context.TRAIN.value }
            _inputs['policy_state_dict'] = policy_state_dict
            ## update algo step
            self.trainer_component.step(_inputs)

        self.trainer_component.post_epoch()

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################
    def evaluate_policy(self):
        assert hasattr(self, "evaluater_component")
        _inputs = {'context': base.Context.EVAL.value }
        for _key in self.components:
            _component = self.components[_key]
            _component.pre_loop(_inputs)

        #self.evaluater_component.pre_loop()

        self.evaluater_component.loop()
        self.evaluater_component.post_loop()

    @property
    def inference_model(self):
        return {
            "actor": self.modules_component.actor,
            "critic": self.modules_component.critic
        }

    @torch.no_grad()
    def get_example_obs(self):

        inputs = {'context': base.Context.EVAL.value}
        self.envwarp_component.pre_loop(inputs)
        self.envwarp_component.pre_step(inputs)
        obs_dict = inputs["obs_dict"]

        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

## for trainer
class PPOTrainer(BasePPO):

    def __init__(self, env: BaseTask, config, log_dir=None, device='cpu'):
        super(PPOTrainer, self).__init__(env, config, log_dir, device)

    @property
    def namespace(self):
        from humanoidverse.agents.ppo import register
        return register.trainer_namespace

## for evaluater
class PPOEvaluater(BasePPO):

    def __init__(self, env: BaseTask, config, log_dir=None, device='cpu'):
        super(PPOEvaluater, self).__init__(env, config, log_dir, device)


    @property
    def namespace(self):
        from humanoidverse.agents.ppo import register
        return register.evaluater_namespace
