import torch
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from hydra.utils import instantiate, get_class
from loguru import logger

# Base class for RL tasks
class BaseTask():
    def __init__(self, config, device):
        self.init_done = False
        ## flags for policy evaluating
        self.config = config
        self.num_envs = self.config.num_envs

        self.debug_viz = False
        self.is_evaluating = False

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # init 1
        ## set simulator
        self._setup_simulator(device)

        ## set terrain
        terrain_mesh_type = self.config.terrain.mesh_type
        self.simulator.setup_terrain(terrain_mesh_type)

        # init 2
        ## instance manager
        self._setup_manager()
        logger.info('instance manager')

        ## load assets
        self.simulator.load_assets()
        logger.info('load_assets')

        # init 3
        ## pre init
        self._pre_init()
        logger.info('_pre_init')

        # init 4
        # create envs, sim and viewer
        self._create_envs()
        logger.info('_create_envs')

        self.simulator.prepare_sim()
        if self.headless == False:
            self.simulator.setup_viewer()

        logger.info('prepare_sim')
        # init 5
        ## init
        self._init()
        logger.info('init')

        # init 6
        ## post init
        self._post_init()
        self.init_done = True
        logger.info('post_init')

    # step
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # step 1
        self._pre_step()

        ## physics_step
        if 'actions' in actor_state:
            actions = actor_state["actions"]
            # stage 2
            self._pre_physics_step(actions)
            self._physics_step()
            self._post_physics_step()
        else:
            self._render()
            for _ in range(self.config.simulator.config.sim.control_decimation):
                self.simulator.simulate_at_each_physics_step()

        self._refresh_sim_tensors()

        ## compute
        # step 2
        self._pre_compute()
        ## check reset flag
        assert hasattr(self, "episode_manager")
        self.episode_manager.compute_reset()
        ## compute reward
        if hasattr(self, "rewards_manager"):
            self.rewards_manager.compute_reward()
        ## reset terminations

        # step 3
        self._reset()

        # set envs
        self._refresh_envs()

        # step 4
        self._compute() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # step 5
        self._post_compute()

        # stage 6
        self._draw_debug_vis()

        items = {}
        if hasattr(self, "observations_manager"):
            items['obs_dict'] = self.observations_manager.obs_buf_dict

        if hasattr(self, "rewards_manager"):
            items['rewards'] = self.rewards_manager.rew_buf

        if hasattr(self, "episode_manager"):
            items['dones'] = self.episode_manager.reset_buf

        if hasattr(self, "extras_manager"):
            items['infos'] = self.extras_manager.extras

        # stage 7
        self._post_step()

        return items


    ###########################################################################
    # Helper functions
    def reset_all(self):
        """ Reset all robots"""
        self._reset(torch.arange(self.num_envs, device=self.device))
        self._refresh_envs()

        if hasattr(self, "actions_manager"):
            actor_state = self.actions_manager.zeros()
            items = self.step(actor_state)
            assert ('obs_dict' in items)
            return items['obs_dict']
        else:
            actor_state = {}
            #actions = torch.zeros(self.num_envs, self.robotdata_manager.num_dof, device=self.device, requires_grad=False)
            #actor_state["actions"] = actions
            return self.step(actor_state)

    def rand_episode_length(self):
        if self.is_evaluating:
            return

        if hasattr(self, "episode_manager"):
            self.episode_manager.rand_episode_length()

    def set_is_evaluating(self):
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True

    # internal functions

    ## init stage
    # init 1
    def _setup_simulator(self, device):
        SimulatorClass = get_class(self.config.simulator._target_)
        self.simulator: BaseSimulator = SimulatorClass(config=self.config, device=device)

        self.headless = self.config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.device = self.simulator.sim_device
        self.sim_dt = self.simulator.sim_dt
        self.up_axis_idx = 2 # Jiawei: HARD CODE FOR NOW

        self.dt = self.config.simulator.config.sim.control_decimation * self.sim_dt

    # init 2
    def _setup_manager(self):
        self.managers = {}
        # HACK for task_data
        _robotdata_manager_class = self.manager_map['robotdata_manager']
        robotdata_manager = _robotdata_manager_class(self)
        robotdata_manager.preinstance_task_data()
        self.managers['robotdata_manager'] = robotdata_manager
        setattr(self, 'robotdata_manager', robotdata_manager)

        for _name, _manager in self.manager_map.items():
            if _name == 'robotdata_manager':
                continue
            _manager = _manager(self)
            self.managers[_name] = _manager
            setattr(self, _name, _manager)

    # init 3
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # env_config = self.config
        self.simulator.create_envs(self.num_envs,
                                    self.terrain_manager.env_origins,
                                    self.robotdata_manager.base_init_state)

    # init 4
    def _pre_init(self):
        for _name,_manager in self.managers.items():
            logger.info(f'pre_init {_name}')
            _manager.pre_init()

    # init 5
    def _init(self):
        for _manager in self.managers.values():
            _manager.init()

        # need _refresh flags
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    # init 6
    def _post_init(self):
        for _manager in self.managers.values():
            _manager.post_init()


    ## step stage
    # step 1
    def _pre_step(self):
        for _manager in self.managers.values():
            _manager.pre_step()

    # physics step
    def _pre_physics_step(self, actions):
        if not hasattr(self, "actions_manager"):
            return
        self.actions_manager.pre_physics_step(actions)

    def _physics_step(self):
        self._render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._physics_step_apply_force()
            self.simulator.simulate_at_each_physics_step()

    def _physics_step_apply_force(self):
        if not hasattr(self, "actions_manager"):
            return
        _actions = self.actions_manager.actual_actions
        self.actuators_manager.pre_physics_step(_actions)
        self.simulator.apply_torques_at_dof(self.actuators_manager.torques)

    def _post_physics_step(self):
        pass

    # compute
    # step 2
    def _pre_compute(self):
        for _manager in self.managers.values():
            _manager.pre_compute()

    # step 3
    def _reset(self, _env_ids = None):
        assert hasattr(self, "episode_manager")
        if _env_ids is None:
            _env_ids = self.episode_manager.reset_env_ids
        else:
            # TODO merger _env_ids & self.episode_manager.reset_env_ids
            pass

        if 0 != len(_env_ids):
            self.need_to_refresh_envs[_env_ids] = True

        # do't check empyt ids for some manager need to update
        for _manager in self.managers.values():
            _manager.reset(_env_ids)

    # step 4
    def _compute(self):
        for _manager in self.managers.values():
            _manager.compute()

    # step 5
    def _post_compute(self):
        for _manager in self.managers.values():
            _manager.post_compute()

    # step 6
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        if self.headless:
            return

        if not self.debug_viz:
            return

        self.simulator.clear_lines()
        self._refresh_sim_tensors()
        for _manager in self.managers.values():
            _manager.draw_debug_vis()

    # step 7
    def _post_step(self):
        for _manager in self.managers.values():
            _manager.post_step()


    ###
    def _render(self, sync_frame_time=True):
        if self.headless:
            return
        self.simulator.render(sync_frame_time)


    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()
        return

    def _refresh_envs(self):
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False

    @property
    def manager_map(self):
        raise Exception("please use sub class")

    @property
    def rewards_map(self):
        raise Exception("please use sub class")
