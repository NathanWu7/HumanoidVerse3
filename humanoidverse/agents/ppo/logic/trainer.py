from agents.base_algo import base
from agents.base_algo import base_algo
import torch

class Trainer(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Trainer, self).__init__(_algo)

        self.clip_param = self.config.clip_param
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef

        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches

    def pre_epoch(self):
        self.loss_dict = {}

    # stage 2
    # level 0

    def step(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs
        assert base.Context.TRAIN.value == _inputs['context']

        # step 1 run policy obtain the new values
        self.algo.modules_component.step(_inputs)

        # step 2 actor loss
        surrogate_loss = self._calcute_surrogate_loss(_inputs)
        self._update_loss('Surrogate', surrogate_loss.item())

        # entropy loss
        entropy_batch = _inputs['entropy_batch']
        entropy_loss = entropy_batch.mean()
        self._update_loss('Entropy', entropy_loss.item())

        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        # critic loss
        value_loss = self._calcute_critic_loss(_inputs)
        self._update_loss('Value', value_loss.item())

        critic_loss = self.value_loss_coef * value_loss

        ## note zero_grad
        # step 2 update rl with kl
        self.algo.optimizer_component.step(_inputs)

        ##
        actor_loss.backward()
        critic_loss.backward()
        self._update_loss('actor', actor_loss.item())
        self._update_loss('critic', critic_loss.item())

        ##
        # model gradient clip step
        self.algo.modules_component.post_step(_inputs)

        ## noptimizer step
        self.algo.optimizer_component.post_step()

        # self.algo.statistics_component.post_step(_inputs)

    def post_epoch(self):
        # NOTE please see rollout
        # clear storage
        #_inputs = {'context': base.Context.TRAIN.value}
        #self.algo.storage_component.pre_loop(_inputs)
        pass

    def _calcute_surrogate_loss(self, _inputs):
        # Surrogate loss
        policy_state_dict = _inputs['policy_state_dict']
        old_actions_log_prob_batch = torch.squeeze(policy_state_dict['actions_log_prob'])
        ##
        actions_log_prob_batch = _inputs['actions_log_prob_batch']

        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)

        ## advantages
        advantages_batch = torch.squeeze(policy_state_dict['advantages'])

        surrogate = -advantages_batch * ratio

        ratio_clamp = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_clipped = -advantages_batch * ratio_clamp

        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss

    def _calcute_critic_loss(self, _inputs):
        # Value function loss
        policy_state_dict = _inputs['policy_state_dict']

        target_values_batch = policy_state_dict['values']
        returns_batch = policy_state_dict['returns']

        value_batch = _inputs['value_batch']

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss

    def _update_loss(self, _key, _value):
        if _key not in self.loss_dict:
            self.loss_dict[_key] = _value
        else:
            self.loss_dict[_key] += _value

    ## help functions for statistics output
    def write(self, writer, _it):
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for loss_key in self.loss_dict.keys():
            writer.add_scalar(f'Loss/{loss_key}', self.loss_dict[loss_key] / num_updates, _it)

    def log_epoch(self, width, pad):
        _buffer = ''
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for loss_key in self.loss_dict.keys():
            _buffer += f"""{loss_key + ' losses:':>{pad}} {self.loss_dict[loss_key] / num_updates:.4f}\n"""
        return _buffer
