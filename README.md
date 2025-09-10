# HumanoidVerse3: A Multi‑Simulator Sim‑to‑Real Framework for Humanoid Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.0.2-silver)](https://isaac-sim.github.io/IsaacLab/)
[![Genesis](https://img.shields.io/badge/Genesis-blue.svg)](https://github.com/Genesis-Embodied-AI/Genesis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

<p align="center">
  <img width="400" src="pics/framework.png" />
  <br/>
  <i>Unified learning and deployment pipeline for humanoid robots.</i>
  <br/>
  <a href="README_zh.md">中文文档</a>
</p>

HumanoidVerse3 is a modular, production‑oriented framework for humanoid robot learning across multiple simulators with a unified API. It supports end‑to‑end workflows from large‑scale reinforcement learning to Sim2Sim verification and real‑world deployment, focusing on whole‑body control and highly dynamic skills.

- Unified abstraction across simulators (Isaac Gym, Isaac Sim, Genesis; MuJoCo for Sim2Sim)
- Task suites for motion tracking and legged locomotion, plus extensible task composition
- Sim2Real pipeline with a unified robot control interface (URCI), ONNX/JIT export, and robustness options (noise, delay, masking)
- Motion processing toolbox (MotionHub) with retargeting for humanoid robots
- Hydra‑based configuration for reproducible experiments at scale

Quick links:
- Chinese README: README_zh.md
- Sim2Sim deployment guide: humanoid_sim2real/README_sim2sim_setup.md

## Features

- Multi‑Simulator Support
  - Isaac Gym for high‑throughput RL on GPU
  - Isaac Sim for photorealistic, high‑fidelity simulation on Omniverse
  - Genesis for modern, high‑precision dynamics
  - MuJoCo as a lightweight Sim2Sim verification environment
- Task Environments
  - Motion tracking from mocap/video to robot joint space
  - Legged base tasks and extensible, modular tasks
- Sim2Real Deployment
  - Unified Robot Control Interface (URCI) for both simulation and hardware
  - ONNX/JIT export for low‑latency inference
  - Robustness injection: noise, delay, masking, IMU perturbations
- Engineering Foundation
  - Modular components, clear separation of concerns
  - Hydra configuration with structured defaults and overrides
  - Integrated logging and experiment tracking (TensorBoard, Weights & Biases)

## System Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA ≥ 11.0 (for GPU acceleration)
- Isaac Gym (Preview 4)
- Isaac Sim 4.5.0 (+ Isaac Lab 2.0.2)
- Genesis 0.2.1

Note: Install simulator‑specific dependencies inside separate environments as recommended below.

## Installation

Create separate conda environments for each simulator to avoid dependency conflicts:

```bash
git clone https://github.com/nathanwu7/HumanoidVerse3.git
cd HumanoidVerse3

# Per‑simulator envs
conda create -n hgym python=3.8     # Isaac Gym
conda create -n hsim python=3.10    # Isaac Sim
conda create -n hgen python=3.10    # Genesis
```

Isaac Gym:

```bash
conda activate hgym

# Download and unpack Isaac Gym Preview 4 (follow NVIDIA instructions/URL)
cd ../
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4

# Python API
pip install -e ./isaacgym/python/

# ld path fix for libpython on env activation
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d ./etc/conda/deactivate.d
cat > ./etc/conda/activate.d/env_vars.sh << 'EOF'
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
EOF
cat > ./etc/conda/deactivate.d/env_vars.sh << 'EOF'
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
unset OLD_LD_LIBRARY_PATH
EOF

# Install HumanoidVerse3
cd -    # back to HumanoidVerse3
pip install -e .
pip install -e humanoidverse/isaac_utils

# Re‑activate to ensure env vars take effect
conda deactivate
conda activate hgym
```

Isaac Sim:

```bash
conda activate hsim
# Follow official Isaac Sim + Isaac Lab installation:
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html

pip install -e .
pip install -e humanoidverse/isaac_utils
```

Genesis:

```bash
conda activate hgen
pip install torch
pip install genesis-world==0.2.1

pip install -e .
pip install -e humanoidverse/isaac_utils
```

## Quick Start

Training (Isaac Gym):

```bash
conda activate hgym
python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking
```

Evaluation and Export:

```bash
python humanoidverse/eval_agent.py \
  +checkpoint=logs/MotionTracking/xxx.pt \
  headless=False
```

During evaluation, ONNX export can be enabled and produces an exported policy beside the checkpoint.

Sim2Sim Demo (MuJoCo):

```bash
python -m humanoid_sim2real.sim2sim.sim2sim
```

## Repository Structure

```
humanoidverse/              # Core framework
  agents/                   # RL algorithms (PPO, multi-head PPO) and analysis callbacks
  envs/                     # Task environments (motion tracking, legged base)
  simulator/                # Simulator adapters (Isaac Gym/Sim, Genesis)
  deploy/                   # URCI abstractions, MuJoCo deployment, policy routing
  utils/                    # Config/logging/inference/motion utilities
humanoid_sim2real/          # Sim2Real configs, Sim2Sim demo, motion library, hardware scripts
MotionHub/                  # Motion processing + SMPL retargeting pipeline
data/                       # Motions and robot assets (e.g., G1, H1)
configs/                    # Hydra configuration (robot/simulator/experiment)
```

## Motion Data

The framework includes diverse motion sets, enabling learning and reproduction of complex dynamic skills.

- PBHC‑based martial arts skills: horse stance + punch, side kick, spinning back kick, hook punch
- Dance motions: Charleston, Bruce Lee–style iconic moves
- MotionHub: end‑to‑end pipeline for video mocap/pose, human‑to‑robot retargeting, physics‑based smoothing

## Design

- Modularity and Abstractions
  - Unified simulator adapters with consistent APIs
  - URCI (URCIRobot) for both simulation and real robots, shared observation/action shaping
  - Robustness options (noise, delay, IMU perturbations) for Sim2Real gap reduction
- Configuration and Logging
  - Hydra configs in `humanoidverse/config` for reproducibility and flexible overrides
  - Integrated Loguru, TensorBoard, W&B for experiment tracking
- Deployment
  - Export policies as TorchScript or ONNX
  - MuJoCo‑based Sim2Sim validation; hardware scripts and configs under `humanoid_sim2real/`

## Documentation

- Sim2Sim deployment: humanoid_sim2real/README_sim2sim_setup.md
- Config references: humanoidverse/config/

If you need additional task‑ or robot‑specific docs, please open an issue or PR.

## GMR Retarget Integration

This repo includes a General Motion Retargeting (GMR) pipeline under `GMR/` for converting SMPL‑X motions to robot joint trajectories.

- Prerequisites
  - Follow `GMR/README.md` to set up GMR and download SMPL‑X body models to `GMR/assets/body_models`.
  - Prepare SMPL‑X motions (e.g., AMASS/OMOMO) in a folder.

- Step 1: Retarget SMPL‑X to your robot (example: Unitree G1)
  ```bash
  python GMR/scripts/smplx_to_robot_dataset.py \
    --robot unitree_g1 \
    --src_folder /path/to/smplx_motions \
    --tgt_folder GMR/out/g1
  ```

- Step 2: Convert GMR outputs to MotionLib format for training
  ```bash
  python -m humanoidverse.utils.gmr.convert_gmr_to_motionlib \
    --src GMR/out/g1 \
    --xml data/robots/g1/g1_29dof_anneal_23dof.xml \
    --out data/motions/GMR/g1/accad_1.pkl
  ```
  - Produces a single training file (dict‑of‑one‑motion) with the fields expected by training and preview.
  - DOF mapping: by default maps Unitree G1 29→23 (removes wrist DOFs). If your source MJCF differs, you can pass `--src-xml` to align by joint names.
  - Interpolation: default 30/30 start/end frames; root rotation is frozen, root position only interpolates Z (XY hold). Set `--interp-start 0 --interp-end 0` to disable.
  - MotionHub parity: appends 3 zero AA joints for extended bodies (hands, head), and computes `contact_mask` via FK; also fills `smpl_joints` for analysis/vis.

- (Optional) Visualize a converted motion
  ```bash
  python -m humanoidverse.utils.gmr.view_converted_motion \
    --motion data/motions/GMR/g1/accad_1.pkl \
    --xml data/robots/g1/g1_29dof_anneal_23dof.xml
  ```
  - The viewer automatically slices extra trailing joints (e.g., the 3 extended bodies) to match the XML motor count.

- Step 3: Point your training config to the converted dataset
  - In `humanoidverse/config/exp/motion_tracking.yaml` or your robot config override, set:
    ```yaml
    robot:
      motion:
        motion_file: data/motions/GMR/g1
    ```
- Then run training as usual (pass a single .pkl file):
  ```bash
  conda activate hgym
  python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking robot.motion.motion_file=data/motions/GMR/g1/accad_1.pkl
  ```

Compatibility
- Training: Converted files expose `fps`, `root_trans_offset`, `pose_aa`, `dof`, `root_rot`, and also include `smpl_joints` and `contact_mask` for MotionHub‑style features.
- Deployment/Sim2Sim: The same MotionLib readers are used in deployment tools; converted files are fully compatible.

### Format Differences: GMR vs MotionLib

- GMR output (per‑motion .pkl)
  - `fps` (int): source motion sampling rate.
  - `root_pos` (T, 3): world translation of the root over time.
  - `root_rot` (T, 4): root orientation as XYZW quaternion over time.
  - `dof_pos` (T, D): scalar joint angles (radians) in the robot DOF order used by GMR.
  - Optional: `local_body_pos`, `link_body_list` (for diagnostics/vis; not required for training).

- MotionLib format (this repo’s training/deploy input)
  - `fps` (int): motion sampling rate (used to determine dt = 1/fps when building states).
  - `root_trans_offset` (T, 3): root world translation used by the environment/task.
  - `pose_aa` (T, 1 + D + 3, 3): axis‑angle representation for root and joints plus 3 extended bodies:
    - `pose_aa[:, 0, :]`: root rotation as a rotation vector (rotvec); converted from `root_rot` (XYZW).
    - `pose_aa[:, 1:1+D, :]`: per‑joint axis‑angle, `axis_i * angle_i`, where `axis_i` is from the MJCF motor’s joint axis.
    - `pose_aa[:, 1+D:, :]`: three zero AA vectors for [left_hand, right_hand, head] extended bodies.
  - `dof` (T, D): scalar joint angles (radians) aligned to the target MJCF motor order.
  - `root_rot` (T, 4): XYZW quaternions for the root.
  - `contact_mask` (T, 2): [left, right] foot contact (velocity + height threshold via FK).
  - `smpl_joints` (T, NB, 3): global joint positions from FK (includes extended bodies when configured).

- Why convert: MotionLib expects unified axis‑angle kinematics across root and joints, and uses robot MJCF to determine joint axes and ordering (motor order). GMR provides quaternion + scalar angles; the converter bridges these conventions.

- Joint ordering and 29→23 mapping (Unitree G1)
  - MotionLib uses the target XML’s motor order; GMR may export 29 DoF including wrists, while training often uses a 23 DoF robot (wrists removed).
  - The converter either:
    - Aligns by joint names when `--src-xml` is provided, dropping missing wrists; or
    - Applies a safe built‑in 29→23 subset for G1 when `--src-xml` is omitted (drops L/R wrist roll/pitch/yaw).

Key meanings inside MotionLib
- `fps`: sets the motion dt used when sampling frames and blending.
- `root_trans_offset`: drives the base position trajectory for the reference motion.
- `pose_aa`: provides root orientation and joint target poses in axis‑angle; the loaders compute FK, velocities and observations from it. The extra 3 joints are no‑ops but align with motion tracking’s extended body pipeline.

Notes
- The converter attempts to compute `contact_mask`/`smpl_joints` via FK. If optional deps are missing, it falls back to defaults (uniform 0.5 contacts, zero joints), which still works for training.

## Citation

If you find this repository useful, please consider citing related prior work and releases:

```bibtex
@misc{HumanoidVerse2,
  author = {Zeng Liangjun},
  title = {HumanoidVerse2: A Multi-Simulator Framework with Modular Design for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/zengliangjun/HumanoidVerse2}},
}

@misc{HumanoidVerse,
  author = {CMU LeCAR Lab},
  title = {HumanoidVerse: A Multi-Simulator Framework for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/LeCAR-Lab/HumanoidVerse}},
}

@article{xie2025kungfubot,
  title={KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills},
  author={Xie, Weiji and Han, Jinrui and Zheng, Jiakun and Li, Huanyu and Liu, Xinzhe and Shi, Jiyuan and Zhang, Weinan and Bai, Chenjia and Li, Xuelong},
  journal={arXiv preprint arXiv:2506.12851},
  year={2025}
}
```

## License

This project is released under the MIT License. See LICENSE for details.

## Contact

Maintainers: Qiwei Wu, Yixiao Feng, Xiangrui Jiang  
Email: nathan.wuqw@gmail.com
