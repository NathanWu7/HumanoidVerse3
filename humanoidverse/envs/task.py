from abc import ABC, abstractmethod

class BaseTask(ABC):

    @abstractmethod
    def reset_all(self):
        pass

    @abstractmethod
    def rand_episode_length(self):
        pass

    @abstractmethod
    def step(self, actor_state: dict):
        '''
        "actions"
        '''
        pass

    def set_is_evaluating(self):
        '''
        Setting Env is evaluating
        '''
