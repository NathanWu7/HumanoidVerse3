# HumanoidVerse2 : A Multi-Simulator Framework with Modular Design for Humanoid Robot Sim-to-Real Learning.

Developed based on [HumanoidVerse ](README_V1.md), please refer to [HumanoidVerse ](README_V1.md) for environment settings and initial operation.  
Aiming to improve the efficiency of robot algorithm development.


# TASK
Extract the task into different submodules, with all submodules having a lifecycle similar to [BaseManager](humanoidverse/envs/base_task/term/base.py)

## BaseManager

<details>
<summary>BaseManager</summary>
```python
class BaseManager():
    def __init__(self, _task):
        self.task = _task                 # task/env object（Sub class of BaseTask）
        self.num_envs = _task.num_envs    # number of environments
        self.device = _task.device        # device of the task/env runing on
        self.config = _task.config        #

    # The first phase, initialization phase
    def pre_init(self):
        pass

    def init(self):
        pass

    def post_init(self):
        pass

    # Phase 2, simulation environment operation phase
    def pre_physics_step(self, actions):
        pass

    def physics_step(self):
        pass

    def post_physics_step(self):
        pass

    # The third stage, the computation phase
    def pre_compute(self):
        pass

    def check_termination(self):
    	'''
        At present, it is only implemented by the episode module for calculating whether the environment terminates
        '''
        pass

    def compute_reward(self):
    	'''
        Currently only implemented by the rewards module for calculating rewards
        '''
        pass

    def reset(self, env_ids):
    	'''
        Used to reset the termination environment
        '''
        pass

    def compute(self):
    	'''
        The related operations of the submodule during the computation phase are implemented here.
        '''
        pass

    def post_compute(self):
        pass

    #The fourth stage is the debug phase
    def draw_debug_vis(self):
        pass
'''

</details>

<details>
<summary>Stages of the lifecycle</summary>

- init: initialization phase
- physics_step:  simulation environment step phase
- compute:  compute phase

</details>

<details>
<summary>dependency situations</summary>

- pre: other submodel depend the submodel
-  no depend(depended)
- post depended other submodel


</details>

<details>
<summary>Special interface</summary>

- check_termination: only implemented by the episode module, used to calculate whether the environment terminates
- compute_reward: only implemented by the rewards module for calculating rewards
</details>


## register

All submodules are registered, created by [BaseTask](humanoidverse/envs/base_task/base_task.py)   
look at [register](humanoidverse/envs/locomotion/term/register.py)

<details>
<summary>module management</summary>

- foundation: the highest level of core submodule
- status:
- assistant:
- mdp:
- sim2real:
- extras:
</details>

  


# AGENT
Extract sub modules from the agent, and the lifecycle of all sub modules is as follows[BaseComponent](humanoidverse/agents/base_algo/base.py)

  

## BaseComponent

<details>

<summary>BaseComponent</summary>

```python
class BaseComponent():
    def __init__(self, _algo: base_algo.BaseAlgo):
        self.algo = _algo                     # agent object（Sub class of BaseAlgo）
        self.num_envs = _algo.env.num_envs    # umber of environments
        self.device = _algo.device            # device of the agent runing on
        self.config = _algo.config            # config

    # The first phase, initialization phase
    def pre_init(self):
        pass

    def init(self):
        pass

    def post_init(self):
        pass

    # Phase 2
    # level 0, the loop
    def pre_loop(self, _inputs = None):
        pass

    def post_loop(self, _inputs = None):
        pass

    # level 1, the epoch
    def pre_epoch(self, _inputs = None):
        pass

    def post_epoch(self, _inputs = None):
        pass

    # level 2, the step
    def pre_step(self, _inputs = None):
        pass

    def step(self, _inputs = None):
        pass

    def post_step(self, _inputs = None):
        pass

    # help function
    def load(self, _loaded_dict: dict):
    	'''
        load from dict
        '''
        pass

    '''
    for saving
    '''
    def update(self, _state_dict: dict):
    	'''
        save to dict
        '''
        pass

    '''
    out the info to writer
    '''
    def write(self, writer, _it):
        pass


    '''
    only for need logging module
    def log_epoch(self, width, pad):
        pass

    '''

```
</details>

<details>

<summary>Stages of the lifecycle</summary>

- init: initialization phase
- loop: level 1, for loop
- epoch: level 2, for epoch
- step: level 2, for step
</details>


## register

All submodules are registered, created by [BaseAlgo](humanoidverse/agents/base_algo/base_algo.py)  
look at [register](humanoidverse/agents/ppo/register.py)
   
module management
   
<details>
<summary>logic</summary>

- evaluater
- rollout
- statistics
- trainer
</details>


<details>
<summary>component</summary>

- envwarp
- modules
- optimizer
- storage
</details>



# Citation
Please use the following bibtex if you find this repo helpful and would like to cite:

```bibtex
@misc{HumanoidVerse2,
  author = {liangjun},
  title = {HumanoidVerse2: A Multi-Simulator Framework with Modular Design for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zengliangjun/HumanoidVerse2}},
}

@misc{HumanoidVerse,
  author = {CMU LeCAR Lab},
  title = {HumanoidVerse: A Multi-Simulator Framework for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LeCAR-Lab/HumanoidVerse}},
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



