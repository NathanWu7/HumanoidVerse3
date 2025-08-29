
import numpy as np
import struct
import copy
import threading
import time
from termcolor import colored
import os
import math
from pprint import pprint
from functools import lru_cache

from typing import Dict, List, Type, Tuple, Any
from dataclasses import dataclass
import dataclasses
from enum import IntEnum


kNumMotors:int = 35
kNumLegMotors:int = 12

#### Pose Array ####

DofPose = Dict[str, Dict[str, Any]]

class PoseArray:
    kNumUsedMotors:int = kNumMotors
    arrorder: List[int] = list(range(kNumUsedMotors))
    
    _joint_names: List[str] = []
    @classmethod
    def __init_joint_names(cls) -> None:
        joint_names = [''] * cls.kNumUsedMotors
        for pname, part in cls._dofpose_specs().items():
            for jname, jid in part.items():
                if jid in cls.arrorder:
                    joint_names[cls.arrorder.index(jid)] = f"{pname}{jname}"
        cls._joint_names = joint_names
    
    @classmethod
    def joint_names(cls, jid: int) -> str:
        if not cls._joint_names:
            cls.__init_joint_names()
        return cls._joint_names[cls.arrorder.index(jid)]
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _dofpose_specs() -> Dict[str, Dict[str, int]]:
        return {
            "LeftLeg": {
                "HipYaw": 0,
                "HipPitch": 1,
                "HipRoll": 2,
                "Knee": 3,
                "AnklePitch": 4,
                "AnkleRoll": 5,
            },
            "RightLeg": {
                "HipYaw": 6,
                "HipPitch": 7,
                "HipRoll": 8,
                "Knee": 9,
                "AnklePitch": 10,
                "AnkleRoll": 11,
            },
            "Waist": {
                "Yaw": 12,
                "Roll": 13,
                "Pitch": 14,
            },
            "LeftArm": {
                "ShoulderPitch": 15,
                "ShoulderRoll": 16,
                "ShoulderYaw": 17,
                "ElbowPitch": 18,
                "WristRoll": 19,
                "WristPitch": 20,
                "WristYaw": 21,
            },
            "RightArm": {
                "ShoulderPitch": 22,
                "ShoulderRoll": 23,
                "ShoulderYaw": 24,
                "ElbowPitch": 25,
                "ElbowRoll": 26,
                "WristPitch": 27,
                "WristYaw": 28,
            },
            "NotUsed": {
                "NotUsed0": 29,
                "NotUsed1": 30,
                "NotUsed2": 31,
                "NotUsed3": 32,
                "NotUsed4": 33,
                "NotUsed5": 34,
            },
        }

    @classmethod
    @lru_cache(maxsize=1)
    def dofpose_specs(cls) -> Dict[str, Dict[str, int]]:
        # filter arrorder
        result = {}
        for leg in cls._dofpose_specs().keys():
            result[leg] = {k: v for k, v in cls._dofpose_specs()[leg].items() if v in cls.arrorder}
        return result

    @classmethod
    def is_valid_array(cls, array: np.ndarray) -> bool:
        return isinstance(array, np.ndarray) and array.shape[0] == cls.kNumUsedMotors
    
    @classmethod
    def is_valid_dofpose(cls, dofpose: DofPose) -> bool:
        if not isinstance(dofpose, dict):
            return False
        spec = cls._dofpose_specs()
        for leg in dofpose.keys():
            if not isinstance(leg, str) or leg not in spec.keys():
                return False
            for joint in dofpose[leg].keys():
                if not isinstance(joint, str) or joint not in spec[leg].keys():
                    return False
        return True

    @classmethod
    def dofpose2array(cls, dofpose: DofPose, dtypezero:Any = 0.):
        assert cls.is_valid_dofpose(dofpose)
        # arr = np.zeros(cls.kNumUsedMotors)
        dtype = type(dtypezero)
        arr = [dtypezero] * cls.kNumUsedMotors
        
        spec = cls.dofpose_specs()
        for leg in spec.keys():
            for joint, idx in spec[leg].items():
                if leg in dofpose and joint in dofpose[leg]:
                    assert isinstance(dofpose[leg][joint], dtype), f"{dofpose[leg][joint],dtype}"
                    arr[cls.arrorder.index(idx)] = dofpose[leg][joint]
        arr = np.array(arr)
        # breakpoint()
        return arr

    @classmethod
    def array2dofpose(cls, array: np.ndarray) -> DofPose:
        assert cls.is_valid_array(array)
        dofpose = {}
        spec = cls.dofpose_specs()
        for leg in spec.keys():
            term = {joint: array[cls.arrorder.index(idx)] for joint, idx in spec[leg].items()}
            if term: dofpose[leg] = term
        return dofpose

    @classmethod
    def expand_array(cls, subcls: Type['PoseArray'], array: np.ndarray) -> np.ndarray:
        """
        Expand the array of `subcls` to the size of `cls`.
        """
        assert subcls.is_valid_array(array)
        assert set(subcls.arrorder) <= set(cls.arrorder)
        result = np.zeros(cls.kNumUsedMotors)
        result[subcls.arrorder] = array
        return result
    
    @classmethod
    def contract_array(cls, subcls: Type['PoseArray'], array: np.ndarray) -> np.ndarray:
        """
        Contract the array of `cls` to the size of `subcls`.
        """
        assert cls.is_valid_array(array)
        assert set(cls.arrorder) >= set(subcls.arrorder)
        full_arr = PoseArray.expand_array(cls, array) 
        result = np.zeros(subcls.kNumUsedMotors)
        result = full_arr[subcls.arrorder]
        return result
    
    @classmethod 
    @lru_cache(maxsize=1)
    def get_symm(cls):
        assert set(cls.arrorder) <= set(Full27DofPoseArray.arrorder)
        symm = Full27DofPoseArray.contract_array(cls,Symmetry)
        
        _perm_full = np.abs(symm).astype(np.int_)
        sign = np.sign(symm)
        perm = np.zeros_like(_perm_full)
        for i in range(len(cls.arrorder)):
            perm[i]=cls.arrorder.index(_perm_full[i])
        return perm, sign
        ...
    
class LeggedPoseArray(PoseArray):
    kNumUsedMotors:int = kNumLegMotors
    arrorder = list(range(kNumLegMotors))
    
class Legged10DofPoseArray(LeggedPoseArray):
    kNumUsedMotors:int = 10
    arrorder = [ 0,1,2,3,4, 
                 6,7,8,9,10,]

# class NHS21DofPoseArray(PoseArray):
#     kNumUsedMotors:int = 21
#     arrorder = [ 0,1,2,3,4,5,
#                  6,7,8,9,10,11,
#                  12,
#                  13,14,15,16,
#                  20,21,22,23
#                 ]

# class Full27DofPoseArray(PoseArray):
#     kNumUsedMotors:int = 27
#     arrorder = list(range(kNumUsedMotors))

# 
# Symmetry = np.array([-6,    7,-8,9,10,-11,
#                      -1e-3, 1,-2,3, 4,- 5,
#                      -12,
#                      20,-21,-22,23,-24,25,-26,
#                      13,-14,-15,16,-17,18,-19
#                      ])

NDof2PoseArray: Dict[int, Type[PoseArray]] = {
    12: LeggedPoseArray,
    10: Legged10DofPoseArray,
    # 21: NHS21DofPoseArray,
    # 27: Full27DofPoseArray,
    35: PoseArray
}


show_27dof = lambda x: pprint(Full27DofPoseArray.array2dofpose(x.cpu().numpy()))

#### Robot Command & State####

URCISTATE = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (q, dq, quat, rpy, omega, gvec, cmd)

@dataclass
class PDCfg:
    kp: np.ndarray
    kd: np.ndarray
    
    def is_valid(self, posecls: Type['PoseArray']) -> bool:
        return posecls.is_valid_array(self.kp) and posecls.is_valid_array(self.kd)
    
    def expand(self, posesubcls: Type['PoseArray'], posecls: Type['PoseArray']) -> 'PDCfg':
        assert self.is_valid(posesubcls) and set(posesubcls.arrorder) <= set(posecls.arrorder)
        return PDCfg(
            kp=posecls.expand_array(posesubcls, self.kp),
            kd=posecls.expand_array(posesubcls, self.kd),
        )
    def contract(self, posesubcls: Type['PoseArray'], posecls: Type['PoseArray']) -> 'PDCfg':
        assert self.is_valid(posecls) and set(posecls.arrorder) >= set(posesubcls.arrorder)
        return PDCfg(
            kp=posecls.contract_array(posesubcls, self.kp),
            kd=posecls.contract_array(posesubcls, self.kd),
        )



class RobotExitException(Exception):
    def __init__(self, message="Safe exit triggered", details=None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message
    pass


if __name__ == "__main__":
    from pprint import pprint

    print(PoseArray.dofpose_specs())
    print(PoseArray.array2dofpose(np.zeros(35)))
    print(PoseArray.joint_names(5))
    
    # print(NHS21DofPoseArray.get_symm())
    print(LeggedPoseArray.get_symm())
    # print(Full27DofPoseArray.get_symm())
