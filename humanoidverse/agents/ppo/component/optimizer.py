from agents.base_algo import base
from agents.base_algo import base_algo
import torch.optim as optim
from loguru import logger
import torch


class Optimizer(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Optimizer, self).__init__(_algo)
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.load_optimizer = self.config.load_optimizer

        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule

    def post_init(self):
        assert hasattr(self.algo, "modules_component")
        modules_component = self.algo.modules_component

        self.actor_optimizer = optim.Adam(modules_component.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(modules_component.critic.parameters(), lr=self.critic_learning_rate)

    def step(self, _inputs):
        self._step_calcute_kl_update_rl(_inputs)
        # merger with pre_step
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def post_step(self):
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    ## help function
    def load(self, loaded_dict: dict):
        if not self.load_optimizer:
            return

        self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
        self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
        #self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate)
        logger.info(f"Optimizer loaded from checkpoint")
        logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
        logger.info(f"Critic Learning rate: {self.critic_learning_rate}")

    def update(self, _dict: dict):
        _items = {
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        _dict.update(_items)

    def write(self, writer, _it):
        writer.add_scalar('Loss/actor_learning_rate', self.actor_learning_rate, _it)
        writer.add_scalar('Loss/critic_learning_rate', self.critic_learning_rate, _it)

    '''
    def set_learning_rate(self, actor_learning_rate, critic_learning_rate):
        ## TODO clear code
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
    '''

    def _step_calcute_kl_update_rl(self, _inputs):
        # update lr with kl
        # KL
        if self.desired_kl == None or self.schedule != 'adaptive':
            return

        policy_state_dict = _inputs['policy_state_dict']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']


        sigma_batch = _inputs['sigma_batch']
        mu_batch = _inputs['mu_batch']

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.e-5) + \
                (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - \
                0.5, axis=-1)

            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.actor_learning_rate
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.critic_learning_rate