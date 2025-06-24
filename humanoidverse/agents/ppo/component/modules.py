from torch import nn
from agents.base_algo import base
from agents.base_algo import base_algo
from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic

class Modules(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Modules, self).__init__(_algo)
        self.algo_obs_dim_dict = self.algo.env.config.robot.algo_obs_dim_dict
        self.num_act = self.algo.env.config.robot.actions_dim
        self.max_grad_norm = self.config.max_grad_norm

    ## help functions
    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def load(self, loaded_dict: dict):
        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])

    def update(self, _state_dict: dict):
        _dict = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
        }
        _state_dict.update(_dict)

    # level 0
    def pre_loop(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs

        ## for learn start
        if base.Context.TRAIN.value == _inputs['context']:
            self.train_mode()
        else:
            self.eval_mode()

    # level 2
    def pre_init(self):
        self.actor = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)

        self.critic = PPOCritic(self.algo_obs_dim_dict,
                                self.config.module_dict.critic).to(self.device)

    def step(self, _inputs):
        ## for ROLLOUT
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs
        if base.Context.ROLLOUT.value == _inputs['context']:
            assert 'obs_dict' in _inputs
            self._step_rollout(_inputs)
        elif base.Context.TRAIN.value == _inputs['context']:
            assert 'policy_state_dict' in _inputs
            self._step_train(_inputs)
        else:
            raise Exception(f"don\'t support now{_inputs['context']}")

    #
    def post_step(self, _inputs = None):
        # inputs is dones info
        assert _inputs is not None
        assert isinstance(_inputs, dict)

        if base.Context.ROLLOUT.value == _inputs['context']:
            assert 'dones' in _inputs
            dones = _inputs['dones']
            self.actor.reset(dones)
            self.critic.reset(dones)
        elif base.Context.TRAIN.value == _inputs['context']:
            # Gradient step
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        else:
            raise Exception(f"don\'t support now{_inputs['context']}")

    ## help functions for statistics output
    def write(self, writer, _it):
        mean_std = self.actor.std.mean().item()
        writer.add_scalar('Policy/mean_noise_std', mean_std, _it)

    def log_epoch(self, width, pad):
        mean_std = self.actor.std.mean().item()
        return f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""

    ##
    def _step_rollout(self, _inputs):
        obs_dict = _inputs['obs_dict']
        policy_state_dict = {}
        policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
        values = self._critic_eval_step(obs_dict).detach()
        policy_state_dict["values"] = values
        _inputs['policy_state_dict'] = policy_state_dict

    ##
    def _step_train(self, _inputs):
        policy_state_dict = _inputs['policy_state_dict']
        actions_batch = policy_state_dict['actions']
        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # record for train
        _inputs['actions_log_prob_batch'] = actions_log_prob_batch
        _inputs['value_batch'] = value_batch
        _inputs['mu_batch'] = mu_batch
        _inputs['sigma_batch'] = sigma_batch
        _inputs['entropy_batch'] = entropy_batch


    # help function
    def _actor_act_step(self, obs_dict):
        return self.actor.act(obs_dict["actor_obs"])

    ## rollout
    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        actions = self._actor_act_step(obs_dict)
        policy_state_dict["actions"] = actions

        action_mean = self.actor.action_mean.detach()
        action_sigma = self.actor.action_std.detach()
        actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        assert len(actions.shape) == 2
        assert len(actions_log_prob.shape) == 2
        assert len(action_mean.shape) == 2
        assert len(action_sigma.shape) == 2

        return policy_state_dict

    ## critic
    def _critic_eval_step(self, obs_dict):
        return self.critic.evaluate(obs_dict["critic_obs"])

