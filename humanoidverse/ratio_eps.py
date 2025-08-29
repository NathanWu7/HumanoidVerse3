import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
import json
# add argparse arguments
import numpy as np
from loguru import logger
import toolz
import joblib


# 1000eps * 350steps w 500 env~~ 3min
# 10000eps * 350steps w 5000 env ~~ 3min + a long time for save files ~~ 6 min ~~ 8GB GPU, 20GB RAM
# 10000eps * 350steps w 10000 env ~~ 3min + a long time for save files ~~ 5 min



def eval_batch_traj(saved_motion_dict, ref_motion_data_preblend, motion_len, tmp_path, robot_cfg_path):
    from humanoidverse.measure_traj import get_appendix_motion_data,get_motionlib_data,blend_motion, eval_accuracy, eval_smoothness
    # for each episode
        # redump one motion to tmp file
        # reload the motion
        # follow the old code to compute the metrics
    # aggregate the metrics
        
    total_result = {
        '_raw': [],
    }
        
    N,L = saved_motion_dict['dof'].shape[0], saved_motion_dict['dof'].shape[1]
    keys_to_save = saved_motion_dict.keys()
    assert L == motion_len, f"Motion length {L} does not match the expected length {motion_len}"
    
    
    for i in range(N):
        dump_data = {}
        
        motion_key = f"motion{i}" 
        dump_data[motion_key] = {
            key: saved_motion_dict[key][i] for key in keys_to_save
        }
        dump_data[motion_key]['fps'] = 50
        
        joblib.dump(dump_data, tmp_path)
        
        
        pol_appendix = get_appendix_motion_data(tmp_path)
        pol_motion_data = get_motionlib_data(tmp_path, robot_cfg_path)
        
        if i ==0:
            ref_motion_data = blend_motion(ref_motion_data_preblend, pol_appendix['motion_times'])

        traj_data = {
            'pol': pol_motion_data,
            'ref': ref_motion_data,
            'appendix': pol_appendix,
        }
    
    
        metrics_accuracy:dict =  toolz.dicttoolz.valmap(lambda x: x.item() * 1e3, eval_accuracy(traj_data,True))
        metrics_smoothness:dict =  toolz.dicttoolz.valmap(lambda x: x.item() * 1e3, eval_smoothness(traj_data,True))
        
        result = {
            'accuracy': metrics_accuracy,
            'smoothness': metrics_smoothness,
        }
        total_result['_raw'].append(result)
        
    # aggregate the metrics
    aggr_accuracy = {}
    aggr_smoothness = {}
    for key in total_result['_raw'][0]['accuracy'].keys():
        key_arr = np.array([total_result['_raw'][i]['accuracy'][key] for i in range(N)])
        aggr_accuracy[key] = {
            'mean': np.mean(key_arr),
            'std': np.std(key_arr),
        }
    for key in total_result['_raw'][0]['smoothness'].keys():
        key_arr = np.array([total_result['_raw'][i]['smoothness'][key] for i in range(N)])
        aggr_smoothness[key] = {
            'mean': np.mean(key_arr),
            'std': np.std(key_arr),
        }
    total_result['accuracy'] = aggr_accuracy
    total_result['smoothness'] = aggr_smoothness
    
    
    return total_result
        
    ...



@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    NoEarlyTermination = False
    NumTotalEps = override_config.get("num_episodes", 1000)
    EpsEvalName = override_config.get("eps_eval_name", "sample_eps")
    
    
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())
    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
    if simulator_type == 'IsaacGym':
        import isaacgym
        
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

    config.headless = True
    pre_process_config(config)

    # use config.device if specified, otherwise use cuda if available
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    if NoEarlyTermination:
        for key in config.env.config.termination:
            if key == 'terminate_when_motion_end': continue
            config.env.config.termination[key] = False
    config.env.config.termination.terminate_when_motion_far = True
    config.env.config.termination_scales.termination_motion_far_threshold = 0.7
    config.env.config.termination_curriculum.terminate_when_motion_far_curriculum = False 

    assert NumTotalEps % config.num_envs == 0, f"NumTotalEps {NumTotalEps} is not divisible by num_envs {config.num_envs}"
    # print(f"config.num_envs: {config.num_envs}"); breakpoint()
    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    config.env.config.save_note="SampleEps"
    config.env.config.enforce_randomize_motion_start_eval=False
    config.robot.motion.motion_lib_type = "WJX"
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent) # commented out for now, might need it back to save motion
    metric_path = (checkpoint.parent / "metrics" / f"ckpt_{ckpt_num}" / (EpsEvalName+"_ratio.json"))
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{config.env.config.ckpt_dir=}")
    print(f"{metric_path=}")
    
    
    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    env = instantiate(config.env, device=device)
    env.config.save_total_steps = ((NumTotalEps/config.num_envs) * env._motion_episode_length).item()
    env._write_to_file = False
    print("num env: ", env.num_envs)
    print("num steps per env: ", env.config.save_total_steps)

    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    ROBOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    algo.evaluate_policy_steps(env.config.save_total_steps+5)
    
    
    saved_motion = env.saved_motion_dict
    print("Motion Sampled")
    
    def calculate_average_first_one(arr):
        # 找到每行第一个 1 的位置，若全 0 则返回 M
        first_one_indices = np.argmax(arr, axis=1)
        # 检查是否有全 0 行：若该行最大值是 0，则全 0
        all_zero_rows = np.max(arr, axis=1) == 0
        # 将全 0 行的位置设为 M
        first_one_indices[all_zero_rows] = arr.shape[1]
        length = np.mean(first_one_indices)
        # 计算平均值
        return length, length / arr.shape[1]
    length, ratio = (calculate_average_first_one(saved_motion['terminate']))
    resdict = {
        'length': length,
        'ratio': ratio,
    }
    with open(str(metric_path), "w") as f:
        json.dump(resdict, f)
    print("Result: ", resdict)
    print(f"Saved Metric Result Dict to {metric_path}")
    


if __name__ == "__main__":
    main()
