
# Grammar: ckpt=_external_xxxx
# Then, ckpt is sent to here `GetExternalPolicy(description)` and get parsed

from humanoidverse.deploy import *
from typing import Dict

def GetZeroPolicy(name: str)->URCIPolicyObs:
    # Usage: $EVALMJC=_external_zero
    import numpy as np
    
    def policy_fn(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        action = np.zeros((1,23))
        return action
        
    def obs_fn(robot: URCIRobot)->np.ndarray:
        # print(robot.timer)
        q_split = np.split(robot.q, [6, 12, 15, 15+4], axis=0)
        print(q_split)
        return np.zeros(1)
        
    return (obs_fn, policy_fn)


def GetSinPolicy(name: str)->URCIPolicyObs:
    # Usage: $EVALMJC=_external_zero
    import numpy as np
    
    action = np.zeros((1,23))
        
    def policy_fn(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        timer = obs_dict['actor_obs'][0]

        # frequency = 
        
        # action[0,3] = np.sin(timer) * 0.3
        # action[0,9] = np.cos(timer) * 0.3
        
        
        
        action[0,4] = np.sin(timer) * 0.5 *4
        action[0,5] = np.cos(timer) * 0.2 *4
        
        action[0,10] = np.cos(timer) * 0.5 *4
        action[0,11] = np.sin(timer) * 0.2 *4
        return action
        
    def obs_fn(robot: URCIRobot)->np.ndarray:
        # print(robot.timer)
        return np.array(robot.timer*robot.dt)
        
    return (obs_fn, policy_fn)

def GetExternalPolicy(description: str, config: CfgType)->URCIPolicyObs:
    external_name = description[len('_external'):]
    
    if external_name.startswith('_zero'):
        return GetZeroPolicy(external_name[5:])
    elif external_name.startswith('_sin'):
        return GetSinPolicy(external_name[4:])
    else:
        raise ValueError(f"Unknown external policy: {external_name}")
    




