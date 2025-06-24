import hydra
import copy
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional
import functools
from hydra._internal.deprecation_warning import deprecation_warning
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import _flush_loggers, configure_log
from hydra.types import TaskFunction
_UNSPECIFIED_: Any = object()
from omegaconf import DictConfig, open_dict, read_write

def _get_rerun_conf(file_path: str, overrides: List[str]) -> DictConfig:
    msg = "Experimental rerun CLI option, other command line args are ignored."
    warnings.warn(msg, UserWarning)
    file = Path(file_path)
    if not file.exists():
        raise ValueError(f"File {file} does not exist!")

    if len(overrides) > 0:
        msg = "Config overrides are not supported as of now."
        warnings.warn(msg, UserWarning)

    with open(str(file), "rb") as input:
        config = pickle.load(input)  # nosec
    configure_log(config.hydra.job_logging, config.hydra.verbose)
    HydraConfig.instance().set_config(config)
    task_cfg = copy.deepcopy(config)
    with read_write(task_cfg):
        with open_dict(task_cfg):
            del task_cfg["hydra"]
    assert isinstance(task_cfg, DictConfig)
    return task_cfg

def main(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[str] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
) -> Callable[[hydra.TaskFunction], Any]:
    """
    :param config_path: The config path, a directory where Hydra will search for
                        config files. This path is added to Hydra's searchpath.
                        Relative paths are interpreted relative to the declaring python
                        file. Alternatively, you can use the prefix `pkg://` to specify
                        a python package to add to the searchpath.
                        If config_path is None no directory is added to the Config search path.
    :param config_name: The name of the config (usually the file name without the .yaml extension)
    """

    hydra.version.setbase(version_base)


    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                args = args_parser.parse_args()
                # FOR DEBUG ONLY
                ## for privileged training
                if "base" == config_name:
                    args.overrides = [
                        '+simulator=isaacgym',
                        '+exp=locomotion',
                        '+domain_rand=NO_domain_rand',
                        '+rewards=loco/reward_h1_locomotion',
                        '+robot=h1/h1_10dof',
                        '+terrain=terrain_locomotion_plane',
                        '+obs=loco/leggedloco_obs_singlestep_withlinvel',
                        'experiment_name=H110dof_loco_IsaacGym',
                        'headless=True',

                        #'num_envs=1',
                        #'project_name=TESTInstallation',

                        'num_envs=4096',
                        'project_name=HumanoidLocomotion',
                        'checkpoint=logs/HumanoidLocomotion/20250411_214128-H110dof_loco_IsaacGym-locomotion-h1_10dof/model_19700.pt'
                    ]
                elif "base_play" == config_name:
                    args.overrides = [
                        '+simulator=isaacgym',
                        '+exp=asap_motion_tracking',
                        '+domain_rand=NO_domain_rand',
                        '+rewards=asap_motion_tracking/reward_motion_tracking_dm_2real',
                        '+robot=g1/g1_29dof_anneal_23dof',
                        '+terrain=terrain_locomotion_plane',
                        '+obs=asap_motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history',
                        'project_name=ASAP',
                        'experiment_name=MotionTracking_CR7',
                        'robot.motion.motion_file=data/motions/asap/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl'
                    ]
                elif "base_eval" == config_name:
                    args.overrides = [
                        '+rewards=loco/reward_g1_locomotion',
                        '+checkpoint=logs/Locomotion/isaacgym/20250517_124306-G123dof_loco_plane-locomotion-g1_29dof_anneal_23dof/model_26000.pt'
                    ]
                else:
                    args.overrides = [

                    ]

                if args.experimental_rerun is not None:
                    cfg = hydra.main._get_rerun_conf(args.experimental_rerun, args.overrides)
                    task_function(cfg)
                    _flush_loggers()
                else:
                    # no return value from run_hydra() as it may sometime actually run the task_function
                    # multiple times (--multirun)
                    _run_hydra(
                        args=args,
                        args_parser=args_parser,
                        task_function=task_function,
                        config_path=config_path,
                        config_name=config_name,
                    )

        return decorated_main

    return main_decorator
