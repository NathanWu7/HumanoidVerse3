import os
import sys
from pathlib import Path

_root = Path(__file__).absolute().parent.parent
print(_root.__str__())
sys.path.append(_root.__str__())

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
from loguru import logger

import threading
from pynput import keyboard

def on_press(key, env):
    try:
       # Force Control
        if hasattr(key, 'char'):
            if key.char == 'N' or key.char == 'n':
                env.next_task()
                logger.info(f"next_task")

    except AttributeError:
        pass

def listen_for_keypress(env):
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()




from utils.config_utils import *  # noqa: E402, F403

@hydra.main(config_path="config", config_name="base_play", version_base="1.1")
def main(config: OmegaConf):
    # import ipdb; ipdb.set_trace()
    #simulator_type = config.simulator['_target_'].split('.')[-1]

    simulator_type = config.simulator.config.name
    # import ipdb; ipdb.set_trace()
    if simulator_type == 'isaacsim45':
        from isaaclab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

    if simulator_type == 'isaacsim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless

        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        # import ipdb; ipdb.set_trace()
    if simulator_type == 'isaacgym':
        import isaacgym  # noqa: F401


    # have to import torch after isaacgym
    import torch  # noqa: E402
    from utils.common import seeding
    import wandb
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    if config.use_wandb:
        project_name = f"{config.project_name}"
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(project=project_name,
                entity=config.wandb.wandb_entity,
                name=run_name,
                sync_tensorboard=True,
                config=unresolved_conf,
                dir=wandb_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pre_process_config(config)

    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    config.env._target_ = config.env._target_ + "Player"
    env = instantiate(config.env, device=device)

    # Start a thread to listen for key press
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()

    env.set_is_evaluating()
    env.reset_all()
    while True:
        _state = {}
        env.step(_state)


if __name__ == "__main__":
    main()
