# PBHC: Physics-Based Humanoid Control Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-red.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3+-green.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

一个用于机器人运动重定向和控制的完整框架，支持从SMPL格式的运动数据到Unitree G1机器人的完整处理流程。

## 🚀 核心功能

- **运动数据处理**: 从视频、AMASS、LAFAN等来源提取SMPL格式运动数据
- **机器人重定向**: 支持Mink和PHC两种重定向算法，将人类运动映射到机器人
- **运动可视化**: 基于MuJoCo的实时运动可视化工具
- **Contact Mask计算**: 自动检测机器人脚部与地面的接触状态
- **RL训练框架**: 基于IsaacGym的强化学习策略训练

## 📋 快速开始

### 环境配置

```bash
# 克隆项目
cd MotioHub

# 安装依赖
conda create -n motionhub python==3.8
pip install -r requirements.txt
```

### 完整工作流程

```bash
# 1. 激活conda环境
conda activate motionhub

# 2. SMPL到机器人重定向
cd smpl_retarget
python mink_retarget/convert_fit_motion.py ../smpl_motion

# 3. 可视化重定向结果
cd ../robot_motion_process
python vis_q_mj.py +motion_file=/path/to/retargeted_motion.pkl

# 4. 计算contact mask
cd ../motion_source
python count_pkl_contact_mask.py robot=unitree_g1_29dof_anneal_23dof +input_folder=/path/to/motion_data

# 5. 运动插值（可选）
cd ../robot_motion_process
python motion_interpolation_pkl.py --origin_file_name=/path/to/input_motion.pkl

# 6. 可视化插值结果
cd robot_motion_process
python vis_q_mj.py +motion_file=/path/to/interpolated_motion.pkl

# 7. 可视化最终结果
cd ../robot_motion_process
python vis_q_mj.py +motion_file=/path/to/motion_with_contact_mask.pkl
```

## 📁 项目结构

```
PBHC/
├── motion_source/              # 运动数据采集和处理
│   ├── demo.py                 # 从视频提取SMPL运动数据
│   ├── count_pkl_contact_mask.py # 计算contact mask
│   └── utils/                  # 工具函数
├── smpl_retarget/              # SMPL到机器人重定向
│   ├── mink_retarget/          # 基于Mink的快速重定向
│   ├── phc_retarget/           # 基于PHC的优化重定向
│   └── poselib/                # 姿态库依赖
├── robot_motion_process/       # 机器人运动处理
│   ├── vis_q_mj.py            # MuJoCo可视化工具
│   └── motion_interpolation_pkl.py # 运动插值
├── smpl_vis/                  # SMPL运动可视化
├── description/               # 机器人和SMPL描述文件
├── example/                   # 示例数据和预训练模型
└── humanoidverse/             # RL训练框架
```

## 🛠️ 详细使用指南

### 1. 运动数据获取
GVHMR
#### 从视频提取运动
```bash

```

#### 处理AMASS数据集
```bash
# 数据会自动保存到smpl_motion文件夹
```

### 2. 运动重定向

#### 使用Mink方法（推荐）
```bash
cd smpl_retarget
python mink_retarget/convert_fit_motion.py /path/to/smpl_motion_folder
```

#### 使用PHC方法
```bash
cd smpl_retarget
python phc_retarget/fit_smpl_motion.py robot=unitree_g1_29dof_anneal_23dof +motion=/path/to/motion_folder
```

### 3. 运动可视化

```bash
cd robot_motion_process
python vis_q_mj.py +motion_file=/path/to/motion.pkl
```

### 4. Contact Mask计算

```bash
cd motion_source
python count_pkl_contact_mask.py robot=unitree_g1_29dof_anneal_23dof +input_folder=/path/to/input_folder
```

## 🔧 配置说明

### 机器人配置
- **Unitree G1**: 支持23DOF和29DOF两种配置
- **配置文件**: `description/robots/cfg/config.yaml`
- **URDF文件**: `description/robots/g1/`

## 📊 输出格式

### 重定向后的数据格式
```python
motion_data = {
    'root_trans_offset': np.array([...]),  # 根节点位置偏移
    'root_rot': np.array([...]),           # 根节点旋转
    'dof': np.array([...]),               # 关节角度
    'fps': 30,                           # 帧率 (Python int)
    'contact_mask': np.array([...])       # 接触掩码 (可选)
}
```

### 文件结构
```
retargeted_motion_data/
├── mink/
│   └── motion.pkl              # Mink重定向结果
└── mink_contact_mask/
    └── motion_cont_mask.pkl    # 带contact mask的数据
```



## 🎨 可视化工具

### SMPL运动可视化

#### Blender可视化
1. 下载Blender 2.9.0版本和SMPL插件
2. 在Blender中添加SMPL对象
3. 运行`import_motion_blender.py`脚本来绑定运动数据

#### PyTorch3D可视化
```bash
python smpl_vis/smpl_render.py --filepath <PATH_TO_MOTION>
```

### 机器人运动可视化

#### MuJoCo可视化（推荐）
```bash
python robot_motion_process/vis_q_mj.py +motion_file=path/to/motion.pkl
```

## ⚙️ 高级工具

### 运动插值
为运动数据添加开始和结束的插值，使其从默认姿态平滑过渡：

```bash
# 基本插值命令
python robot_motion_process/motion_interpolation_pkl.py --origin_file_name=path/to/motion.pkl --start=0 --end=100 --start_inter_frame=30 --end_inter_frame=30

# 可视化插值结果
python robot_motion_process/vis_q_mj.py +motion_file=path/to/interpolated_motion.pkl

# 示例：可视化的插值结果
python robot_motion_process/vis_q_mj.py +motion_file=/home/jxr/HumanoidVerse3/data/motions/GVHMR/Khalil/love_song_inter0.5_S0-30_E166-30.pkl
```

### 轨迹分析
使用`traj_vis.ipynb`笔记本分析运动轨迹，可以比较仿真轨迹与参考运动。

## 📚 核心库说明

### poselib库
poselib是一个用于加载、操作和重定向骨骼姿势和运动的库：

- **poselib.core**: 基础数据加载和张量操作
- **poselib.skeleton**: 高级骨骼操作和重定向
- **poselib.visualization**: 骨骼可视化

### SMPL-Sim模拟器
SMPL-Sim支持在MuJoCo和Isaac Gym中创建SMPL兼容的人体模型：

```bash
# 运行示例环境
python examples/env_humanoid_test.py headless=False

# 训练策略
python smpl_sim/run.py env=speed exp_name=speed env.self_obs_v=2
```

## 🐛 常见问题

### 1. fps值类型问题
**问题**: 生成的pkl文件中的fps是numpy数组类型
**解决**: 已在最新版本中修复，fps现在保存为Python整数类型

### 2. 运动质量不佳
**建议**:
- 使用`--correct`参数启用运动矫正
- 检查输入的SMPL数据质量
- 调整重定向参数

### 3. 可视化窗口不显示
**检查**:
- 确保MuJoCo正确安装
- 检查图形驱动程序
- 尝试使用虚拟显示: `export DISPLAY=:0`

## 📝 更新日志

- **v1.0.0** (2025-01): 初始版本发布
  - 支持Mink和PHC重定向算法
  - 集成运动可视化工具
  - 添加contact mask计算
  - 修复fps类型问题

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可证。

## 🙏 致谢

- [MaskedMimic](https://github.com/NVlabs/ProtoMotions): 重定向算法基础
- [PHC](https://github.com/ZhengyiLuo/PHC): 优化重定向算法
- [GVHMR](https://github.com/zju3dv/GVHMR): 视频运动提取
- [Unitree](https://www.unitree.com/): G1机器人支持
