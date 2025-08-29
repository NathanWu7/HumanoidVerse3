
# Policy

This directory contains the source code for policy training and evaluation for PBHC. We also provide some pretrained policies for the `Horse-stance pose` motion ( at `PBHC/code/policy/example/pretrained_horse_stance_pose`).

Our code is based on the [ASAP [1]](https://github.com/LeCAR-Lab/ASAP) official codebase. Note that there are some experimental features in this codebase, which are not used anywhere. Some of them are potentially useful but we cannot guarantee the correctness and stability.

[1] ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills 







## Policy Training
To train a policy, you can use the following command.
- Change the `robot.motion.motion_file` in the command below to the motion you want to train. Sample motion datasets are provided in `example/motion_data/`.
- Change the `num_envs`. We set it to `4096` for training, but you can set it to `128` for debugging.
- The output policy ckpt will be saved in `logs/MotionTracking/` with the name format `YYYYMMDD_HHMMSS-debug-motion_tracking-g1_23dof_lock_wrist` (e.g. `20990521_180647-debug-motion_tracking-g1_23dof_lock_wrist`).
- We train `50000` iterations for each experiment in the paper.


```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=debug \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0
```

You can also use the following command for benchmark a new motion data. This experiment setting takes no domain randomization and includes privilege information in actor observation. Hence it's not deployable, but can be used as an oracle for checking the motion quality and difficulty.


```bash
python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=128 \
+obs=motion_tracking/benchmark \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=dr_nil \
+rewards=motion_tracking/main \
experiment_name=benchmark \
robot.motion.motion_file="example/motion_data/Horse-stance_pose.pkl" \
seed=1 \
+device=cuda:0
```


## Policy Evaluation
The following commands provide examples of how to evaluate the trained policy.
- `eval_agent.py`: run the policy with visualization in IsaacGym.
- `sample_eps.py`: rollout the policy for several episodes, output the evaluation metrics (accuracy and smoothness). The early termination mechanism is disabled, so the agent will run from the beginning to the end of the motion. 
  - Change the `num_envs` and `num_episodes` to the number of episodes you want to evaluate, these two should be the same.
- `ratio_eps.py`: rollout the policy for several episodes, output the mean episode length and the ratio of the mean episode length to the reference motion length. The early termination mechanism is enabled, so the agent will stop when the motion is finished for computing the episode length. 
  - Same usage as the above.

```bash

python humanoidverse/eval_agent.py +device=cuda:0 +env.config.enforce_randomize_motion_start_eval=False +checkpoint=example/pretrained_horse_stance_pose/model_50000.pt

python humanoidverse/sample_eps.py +device=cuda:0  +checkpoint=example/pretrained_horse_stance_pose/model_50000.pt +num_envs=1 +num_episodes=1 +eps_eval_name=samtraj +opt=record

python humanoidverse/ratio_eps.py +device=cuda:0 +checkpoint=example/pretrained_horse_stance_pose/model_50000.pt +opt=record +num_envs=100 +num_episodes=100 +eps_eval_name=example
```

##  Deployment
To deploy the policy in MuJoCo, you need to:
- First export the policy to onnx by running the `eval_agent.py` above, the exported policy appears in the `exported` folder. We have provided the exported policy for the pretrained checkpoint in `example/pretrained_horse_stance_pose/exported/model_50000.onnx`.
- Then run the following command to deploy the policy in MuJoCo.

  ```bash
  python humanoidverse/urci.py +opt=record +simulator=mujoco +checkpoint=example/pretrained_horse_stance_pose/exported/model_50000.onnx
  ```

- See `class MViewerPlugin` in `humanoidverse/deploy/mujoco.py` for the key mapping.
- You can also deploy multiple policies in one run. Press `0` to select the first policy, `1` to select the second policy, etc. You can also include a locomotion policy to make the robot move, see `humanoidverse/deploy/external/core.py` to add your own policy.

  ```bash
  python humanoidverse/urci.py +checkpoint='[example/pretrained_horse_stance_pose/exported/model_50000.onnx,example/pretrained_horse_stance_pose/exported/model_50000.onnx]' +opt=record +simulator=mujoco
  ```

To deploy the policy in real-world robot, you need to:
- Run the Sim2sim deployment first for testing.
- Write a real-world deployment module for the robot, same interface as the mujoco module, see `humanoidverse/deploy/mujoco.py` and `humanoidverse/deploy/urcirobot.py`.
- Ensure the **SAFETY** of both humans and hardware:
  - Test in a controlled environment with safety measures in place (e.g., emergency stop button, safe zone barriers).
  - Verify joint limits, velocity/acceleration caps, and torque constraints in the control code before execution.
  - Use slow motion or small torque settings during initial trials.
