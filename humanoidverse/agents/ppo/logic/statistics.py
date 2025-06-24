from agents.base_algo import base
from agents.base_algo import base_algo
from collections import deque
from humanoidverse.utils.average_meters import TensorAverageMeterDict
import torch
import statistics
from collections import deque
from rich.panel import Panel
from rich.live import Live
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from rich.console import Console
console = Console()

class Statistics(base.BaseComponent):
    def __init__(self, _algo: base_algo.BaseAlgo):
        super(Statistics, self).__init__(_algo)

        self.num_steps_per_env = self.config.num_steps_per_env
        self.num_learning_iterations = self.config.num_learning_iterations

        self.tot_timesteps = 0
        self.tot_time = 0
        self.log_dir = self.algo.log_dir


    def init(self):
        # Book keeping
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.episode_env_tensors = TensorAverageMeterDict()
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

    ## start epoch
    def pre_epoch(self, _inputs = None):
        self.ep_infos.clear()

    def post_step(self, _inputs):
        ## for ROLLOUT
        assert _inputs is not None
        assert isinstance(_inputs, dict)
        if base.Context.ROLLOUT.value == _inputs['context']:
            self._post_step_rollout(_inputs)
        #if base.Context.TRAIN.value == _inputs['context']:
        else:
            raise Exception('don\'t implement this operator')

    def post_epoch(self, log_dict):
        log_dict['ep_infos'] = self.ep_infos
        log_dict['rewbuffer'] = self.rewbuffer
        log_dict['lenbuffer'] = self.lenbuffer

        self._post_epoch_logging(log_dict)

    # help step
    def _post_step_rollout(self, _inputs):

        # final process
        dones = _inputs['dones']
        rewards = _inputs['rewards']
        infos = _inputs['infos']
        self.episode_env_tensors.add(infos["to_log"])

        if self.log_dir is not None:
            # Book keeping
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])
            self.cur_reward_sum += rewards
            self.cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            if 0 == len(new_ids):
                return

            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        ep_string = f''
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        train_log_dict = {}
        fps = int(self.num_steps_per_env * self.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))
        train_log_dict['fps'] = fps

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        self._logging_to_writer(log_dict, train_log_dict, env_log_dict)

        #if hasattr(self.algo, "trainer_component"):
        str = f" \033[1m Learning iteration {log_dict['it']}/{self.algo.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict[
                            'collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (collection: {log_dict[
                            'collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                        )

        for _key in self.algo.components:
            _component = self.algo.components[_key]
            if hasattr(_component, "log_epoch"):
                log_string += _component.log_epoch(width, pad)

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (
                               log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        # Use rich Live to update a specific section of the console
        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            # Your training loop or other operations
            pass

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        # Logging Loss Dict
        for _key in self.algo.components:
            self.algo.components[_key].write(self.writer, log_dict['it'])

        self.writer.add_scalar('Perf/total_fps', train_log_dict['fps'], log_dict['it'])
        self.writer.add_scalar('Perf/collection time', log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])
