from agents.base_algo import base
from agents.base_algo import base_algo
from humanoidverse.agents.modules.data_utils import RolloutStorage
import torch

class Storage(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Storage, self).__init__(_algo)
        self.num_steps_per_env = self.config.num_steps_per_env
        self.algo_obs_dim_dict = self.algo.env.config.robot.algo_obs_dim_dict
        self.num_act = self.algo.env.config.robot.actions_dim

        self.gamma = self.config.gamma
        self.lam = self.config.lam

        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches

    # stage 1
    def pre_init(self):
        self.storage = RolloutStorage(self.num_envs, self.num_steps_per_env)
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)

        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('rewards', shape=(1,), dtype=torch.float)
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool)
        self.storage.register_key('values', shape=(1,), dtype=torch.float)
        self.storage.register_key('returns', shape=(1,), dtype=torch.float)
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)

    def pre_epoch(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs

        if base.Context.ROLLOUT.value == _inputs['context']:
            self.storage.clear()
        elif base.Context.TRAIN.value == _inputs['context']:
            _inputs['generator'] = self.storage.mini_batch_generator(self.config.num_mini_batches, \
                                                                 self.config.num_learning_epochs)
        else:
            raise Exception(f"don\'t support now{_inputs['context']}")

    # stage 3
    '''
    obs_dict, policy_state_dict
    '''
    def step(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs
        if base.Context.ROLLOUT.value == _inputs['context']:
            self._step_rollout(_inputs)
        else:
            raise Exception(f"don\'t support now{_inputs['context']}")

    '''
    rewards, dones
    '''
    def post_step(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs
        if base.Context.ROLLOUT.value == _inputs['context']:
            self._post_step_rollout(_inputs)
        else:
            raise Exception(f"don\'t support now{_inputs['context']}")

    '''
    compute returns, advantages
    '''
    def post_epoch(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs

        if base.Context.ROLLOUT.value == _inputs['context']:
            assert 'obs_dict' in _inputs
            returns, advantages = self._compute(_inputs)

            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        elif base.Context.TRAIN.value == _inputs['context']:
            self.storage.clear()
        else:
            raise Exception(f"don\'t support now{_inputs['context']}")


    ## help functions
    def _step_rollout(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'obs_dict' in _inputs
        assert 'policy_state_dict' in _inputs

        ## Append states to storage
        obs_dict = _inputs['obs_dict']
        for obs_key in obs_dict.keys():
            self.storage.update_key(obs_key, obs_dict[obs_key])

        policy_state_dict = _inputs['policy_state_dict']
        for obs_ in policy_state_dict.keys():
            self.storage.update_key(obs_, policy_state_dict[obs_])

    def _post_step_rollout(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'rewards' in _inputs
        assert 'dones' in _inputs
        assert 'policy_state_dict' in _inputs

        rewards = _inputs['rewards']
        dones = _inputs['dones']
        infos = _inputs['infos']
        policy_state_dict = _inputs['policy_state_dict']

        rewards_stored = rewards.clone().unsqueeze(1)
        if 'time_outs' in infos:
            rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
        assert len(rewards_stored.shape) == 2
        self.storage.update_key('rewards', rewards_stored)
        self.storage.update_key('dones', dones.unsqueeze(1))
        self.storage.increment_step()

    def _compute(self, _inputs):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the
        environment based on the provided observations and policy state. It uses
        Generalized Advantage Estimation (GAE) to compute the advantages, which
        helps in reducing the variance of the policy gradient estimates.

            last_obs_dict (dict): The last observation dictionary containing the
                      final state of the environment.
            policy_state_dict (dict): A dictionary containing the policy state
                          information, including 'values', 'dones',
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        last_obs_dict = _inputs['obs_dict']
        last_values= self.algo.modules_component.critic.evaluate(last_obs_dict["critic_obs"]).detach()
        advantage = 0

        ###
        values = self.storage.query_key('values')
        dones = self.storage.query_key('dones')
        rewards = self.storage.query_key('rewards')

        ###
        last_values = last_values.to(self.device)

        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        ## return
        returns = torch.zeros_like(values)

        num_steps = returns.shape[0]

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]

            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

