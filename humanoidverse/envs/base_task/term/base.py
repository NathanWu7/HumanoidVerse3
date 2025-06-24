class BaseManager():
    def __init__(self, _task):
        '''
        simulator had setup
        '''
        self.task = _task
        self.num_envs = _task.num_envs
        self.device = _task.device
        self.config = _task.config
        self.is_initialized = False
        if hasattr(self.task, "task_data"):
            self.task_data = self.task.task_data

    # stage 1
    def pre_init(self):
        '''
        help for simulator had loaded assets
        and waitting for create envs
        '''
        pass

    def init(self):
        '''
        simulator ready, applicable to initialization of each manager that does not depend on other managers
        '''
        pass

    def post_init(self):
        '''
        simulator ready, applicable to initialization of each manager that depend on other managers
        '''
        pass

    # stage 2
    def pre_step(self):
        '''
        task pre step, used for clean history invalid info
        '''
        pass


    def pre_physics_step(self, actions):
        # actions, actuators
        pass

    def physics_step(self):
        pass

    def post_physics_step(self):
        pass

    # stage 3
    # compute
    def pre_compute(self):
        pass

    def compute_reset(self):
        # only for episode
        # update to compute
        pass

    def compute_reward(self):
        # only for ewards
        # update to compute
        pass

    def reset(self, env_ids):
        pass

    def compute(self):
        pass

    def post_compute(self):
        pass

    # stage 4
    def draw_debug_vis(self):
        pass

    def post_step(self):
        pass