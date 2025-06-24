## Commands
  
```bash
################ Example Train ###############
python humanoidverse/train_agent.py \
+simulator=<simulator_name> \
+exp=<task_name> \
+domain_rand=<domain_randomization> \
+rewards=<reward_function> \
+robot=<robot_name> \
+terrain=<terrain_name> \
+obs=<observation_name> \
num_envs=<num_envs> \
project_name=<project_name> \
experiment_name=<experiment_name> \
headless=<headless_mode>
# +opt=wandb

##############  Example Eval #################
python humanoidverse/eval_agent.py +checkpoint=logs/xxx/../xx.pt \
+simulator=<simulator_name>  # sim2sim

############ Example Sim2real #########
python3 humanoid_sim2real/deployment_scripts/your_deployment_script.py your_ethernet
```

## G1 Examples
```bash
###########   Train   ###### G1 #######   Locomotion ##### isaacsim 4.5 ######
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacsim45 \
+exp=locomotion \
+domain_rand=domain_rand_base \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=Locomotion \
experiment_name=G123dof_loco_plane_domain_rand \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.00003


###############   Train   ##### G1 #####   Motion tracking ##### isaacsim 4.5 #######
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacsim45 +exp=asap_motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=asap_motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=asap_motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history project_name=ASAP \
experiment_name=MotionTracking \
num_envs=4096 \   #for genesis num_env=1024
robot.motion.motion_file=data/motions/PBHC/motion_data/Bruce_Lee_pose.pkl \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0
```

## Motions
```bash
################  motion test  ######  genesis ########
HYDRA_FULL_ERROR=1 python humanoidverse/play_agent.py \
+simulator=genesis \
+exp=asap_motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=asap_motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=asap_motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
project_name=ASAP \
experiment_name=MotionTracking \
robot.motion.motion_file=data/motions/PBHC/motion_data/Bruce_Lee_pose.pkl 
```
TODO motion_convert_tools

```bash
############ eval ###### isaacsim45 to genesis (sim 2 sim) ###############
HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
+checkpoint=logs/ASAP/isaacsim45/20250623_161051-MotionTracking_test-motion_tracking-g1_29dof_anneal_23dof/model_1000.pt \ 
+simulator=genesis
```

## Policy Deployment
```bash
############ eval ###### sim 2 real ######## my settings#######
python3 humanoid_sim2real/deployment_scripts/hardware_whole_body_G1_23dof_kungfu.py enp1s0
```