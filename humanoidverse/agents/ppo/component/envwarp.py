from agents.base_algo import base
from agents.base_algo import base_algo

class EnvWarp(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(EnvWarp, self).__init__(_algo)

        self.init_at_random_ep_len = self.config.init_at_random_ep_len

    # level 0
    def pre_loop(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'context' in _inputs

        ## for learn start
        if base.Context.TRAIN.value == _inputs['context']:
            # setup train model
            if self.init_at_random_ep_len:
                self.algo.env.rand_episode_length()
        else:
            # setup eval model
            self.algo.env.set_is_evaluating()

        obs_dict = self.algo.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        self.obs_dict = obs_dict

    # level 2
    def pre_step(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert self.obs_dict is not None

        _inputs['obs_dict'] = self.obs_dict

    def step(self, _inputs):
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        assert 'policy_state_dict' in _inputs
        # check context ?
        policy_state_dict = _inputs['policy_state_dict']

        actions = policy_state_dict["actions"]
        actor_state = {}
        actor_state["actions"] = actions
        _items = self.algo.env.step(actor_state)
        ##

        obs_dict = _items['obs_dict']
        dones = _items['dones']
        infos = _items['infos']
        # critic_obs = privileged_obs if privileged_obs is not None else obs
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        dones = dones.to(self.device)


        if 'rewards' in _items:
            rewards = _items['rewards']
            rewards = rewards.to(self.device)
            _inputs['rewards'] = rewards

        _inputs['obs_dict'] = obs_dict # TODO don't update
        _inputs['dones'] = dones
        _inputs['infos'] = infos

        self.obs_dict = obs_dict # update
