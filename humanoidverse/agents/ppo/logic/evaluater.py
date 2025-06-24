from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from agents.base_algo import base
from agents.base_algo import base_algo
from hydra.utils import instantiate
import torch

class Evaluater(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Evaluater, self).__init__(_algo)

        self.eval_callbacks: list[RL_EvalCallback] = []

    def init(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))


    def pre_loop(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs
        assert _inputs['context'] == base.Context.EVAL.value
        # callback pre
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

        self.eval_policy = self.algo.modules_component.actor.act_inference
        # env eval flag

        # init work contexts info
        self.inputs = {
            'context': base.Context.EVAL.value,
            "done_indices": [],
            "stop": False
        }

    def post_loop(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def step(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs
        assert _inputs['context'] == base.Context.EVAL.value

        # stage 1
        # 1 feature obs_dict
        self.algo.envwarp_component.pre_step(_inputs)

        # 2
        actions = self.eval_policy(_inputs["obs_dict"]['actor_obs'])
        _policy_dict = {"actions": actions}
        _inputs.update({'policy_state_dict': _policy_dict})
        # 2 callbacks
        for c in self.eval_callbacks:
            _inputs = c.on_pre_eval_env_step(_inputs)

        # 3
        self.algo.envwarp_component.step(_inputs)

        # 3 callbacks
        for c in self.eval_callbacks:
            _inputs = c.on_post_eval_env_step(_inputs)

    @torch.no_grad()
    def loop(self):
        step = 0

        while True:
            self.inputs["step"] = step
            self.step(self.inputs)

            step += 1
