# HumanoidVerse3：多仿真器人形机器人仿真到真实学习框架

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.0.2-silver)](https://isaac-sim.github.io/IsaacLab/)
[![Genesis](https://img.shields.io/badge/Genesis-blue.svg)](https://github.com/Genesis-Embodied-AI/Genesis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

<p align="center">
  <img width="400" src="pics/framework.png">
</p>

HumanoidVerse3是一个先进的人形机器人仿真到真实学习框架，支持多种仿真环境和机器人平台。该框架基于模块化设计，支持便捷地在不同仿真器之间切换，并提供完整的从训练到部署的解决方案。

[📖 English Documentation](README.md) | [🚀 快速开始](#快速开始) | [📚 详细文档](README_V2.md)

## ✨ 核心特性

### 🔧 多仿真器支持
- **Isaac Gym** - NVIDIA高性能GPU并行仿真
- **Isaac Sim** - 基于Omniverse的先进仿真平台
- **Genesis** - 现代物理引擎仿真器
- **MuJoCo** - 灵活的刚体动力学仿真

### 🤖 多种机器人平台
- **Unitree G1** - 29DOF/23DOF/12DOF配置
- **Unitree H1** - 19DOF/10DOF配置
- **自定义机器人** - 模块化设计支持扩展

### 🎯 丰富任务支持
- **运动跟踪** - 基于视频的运动捕捉和重现
- **武术技能** - 马步冲拳、侧踢、后旋踢等动态动作
- **舞蹈动作** - Charleston舞等复杂动作序列
- **自定义任务** - 模块化架构便于扩展

### 🚀 Sim2Real能力
- **硬件部署** - 完整的机器人硬件接口
- **运动转换** - 从视频到机器人动作的转换
- **实时控制** - ONNX模型推理优化

## 📋 环境要求

| 组件 | 版本要求 | 用途 |
|------|---------|------|
| Python | ≥3.8 | 核心运行时 |
| PyTorch | ≥1.12 | 深度学习框架 |
| CUDA | ≥11.0 | GPU加速 |
| Isaac Gym | Preview 4 | NVIDIA仿真 |
| Isaac Sim | 4.5.0 | Omniverse仿真 |
| Genesis | 0.2.1 | 现代物理仿真 |

## 🛠️ 安装指南

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/nathanwu7/HumanoidVerse3.git
cd HumanoidVerse3

# 为每个仿真器创建独立的conda环境
conda create -n hgym python=3.8    # IsaacGym环境
conda create -n hsim python=3.10   # IsaacSim环境
conda create -n hgen python=3.10   # Genesis环境
```

### 2. 安装Isaac Gym

```bash
conda activate hgym

# 下载Isaac Gym
cd ../
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4

# 安装Python API
pip install -e ./isaacgym/python/

# 配置环境变量以处理共享库问题ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

# 激活环境时自动设置LD_LIBRARY_PATH
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
cat > ./etc/conda/activate.d/env_vars.sh << 'EOF'
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
EOF

# 退出环境时恢复LD_LIBRARY_PATH
mkdir -p ./etc/conda/deactivate.d
cat > ./etc/conda/deactivate.d/env_vars.sh << 'EOF'
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}
unset OLD_LD_LIBRARY_PATH
EOF

# 安装HumanoidVerse3
cd HumanoidVerse3
pip install -e .

# 重要：重新激活环境以使环境变量生效
conda deactivate
conda activate hgym
```

### 3. 安装Isaac Sim

```bash
conda activate hsim

# 按照官方文档安装Isaac Sim和Isaac Lab
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html

# 安装HumanoidVerse3
pip install -e .
```

### 4. 安装Genesis

```bash
conda activate hgen

# 安装Genesis
pip install torch
pip install genesis-world==0.2.1

# 安装HumanoidVerse3
pip install -e .
```

## 🚀 快速开始

### 训练示例

```bash
# 确保在Isaac Gym环境中
conda activate hgym

# Isaac Gym环境训练运动跟踪任务
python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking
```

### 评估示例

```bash
# 加载训练好的模型进行评估
python humanoidverse/eval_agent.py \
  +checkpoint=logs/MotionTracking/xxx.pt \
  headless=False
```

### Sim2Real部署

```bash
# 启动MuJoCo仿真验证
python -m humanoid_sim2real.sim2sim.deploy_mujoco_sim_kungfu
```

## 📁 项目结构

```
HumanoidVerse3/
├── humanoidverse/              # 核心框架代码
│   ├── agents/                 # 强化学习算法
│   │   ├── ppo/               # PPO算法实现
│   │   └── mh_ppo/            # 多头PPO算法
│   ├── envs/                  # 环境定义
│   │   ├── motion_tracking/   # 运动跟踪任务
│   │   └── legged_base_task/  # 基础腿部控制任务
│   ├── simulator/             # 仿真器接口
│   │   ├── isaacgym/         # IsaacGym接口
│   │   ├── isaacsim/         # IsaacSim接口
│   │   └── genesis/          # Genesis接口
│   └── utils/                # 工具函数
├── humanoid_sim2real/         # Sim2Real部署工具
│   ├── ckpt_demo/            # 预训练模型
│   ├── configs/              # 部署配置
│   └── motion_lib/           # 运动库
├── data/                     # 数据资源
│   ├── motions/              # 运动数据
│   │   ├── PBHC/            # 武术动作数据
│   │   └── GVHMR/           # 通用运动数据
│   └── robots/               # 机器人模型
│       ├── g1/               # G1机器人模型
│       └── h1/               # H1机器人模型
└── configs/                  # Hydra配置文件
    ├── robot/                # 机器人配置
    ├── simulator/            # 仿真器配置
    └── exp/                  # 实验配置
```

## 🎯 运动数据

项目包含丰富的运动数据，支持多种动态动作：

### 武术动作 (PBHC数据集)
- **马步冲拳** - Kung Fu基本功动作
- **侧踢** - 高难度踢击动作
- **后旋踢** - 复杂旋转动作
- **钩拳** - 快速打击动作

### 舞蹈动作
- **Charleston舞** - 复古舞蹈动作
- **Bruce Lee姿势** - 李小龙标志性动作

### 运动转换工具
- 支持从视频提取运动数据
- 自动转换为机器人关节角度
- 基于PBHC框架的先进算法

## 🔬 研究特色

### 模块化架构
- **BaseManager** - 统一的生命周期管理
- **BaseComponent** - 算法组件模块化
- **Hydra配置** - 灵活的参数管理

### 多仿真器一致性
- 统一的API接口设计
- 自动参数转换和适配
- 跨仿真器模型迁移

### 高级运动控制
- 基于物理的整体控制
- 动态技能学习能力
- 实时优化算法

## 📖 使用文档

详细的使用说明请参考：

- [训练指南](README_V2.md) - 完整的训练流程
- [Sim2Real部署](humanoid_sim2real/README_sim2sim_setup.md) - 部署指南
- [配置说明](humanoidverse/config/) - 参数配置详解

## 🔗 相关工作

本项目基于以下优秀工作：

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

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📧 联系我们

- 项目维护者: Qiwei Wu, Yixiao Feng
- 邮箱: nathan.wuqw@gmail.com

---
