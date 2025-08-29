# HumanoidVerse Sim2Real - Sim2Sim 启动指南

## 环境准备

### 1. 安装依赖
```bash
cd /home/jxr/HumanoidVerse3/humanoid_sim2real
pip install -r requirements.txt
```

### 2. 激活conda环境（如果使用conda）
```bash
conda activate your_env_name  # 替换为你的环境名
```

## 启动Sim2Sim验证

### 基本启动命令
```bash
# 方法1: 直接在项目根目录运行
cd /home/jxr/HumanoidVerse3
python -m humanoid_sim2real.sim2sim.deploy_mujoco_sim_kungfu

# 方法2: 在sim2sim目录中运行
cd /home/jxr/HumanoidVerse3/humanoid_sim2real/sim2sim
python sim2sim.py
```

### 配置文件说明
默认配置文件位置：`humanoid_sim2real/configs/g1_ref_kungfu.yaml`

配置文件包含：
- **模型路径**: ONNX推理模型 `horse_stance_punch_model_33000.onnx`
- **动作数据**: 马步冲拳动作 `Horse-stance_punch.pkl`
- **机器人模型**: G1机器人29DOF模型 `g1_29dof_anneal_23dof_fitmotionONLY.xml`
- **控制参数**: PD控制器参数、关节限制等

### 验证步骤
1. **环境检查**: 确保所有依赖已安装
2. **文件检查**: 确认以下文件存在：
   - `data/motions/PBHC/motion_data/Horse-stance_punch.pkl`
   - `humanoid_sim2real/ckpt_demo/horse_stance_punch_model_33000.onnx`
   - `data/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml`
3. **启动仿真**: 运行上述命令之一
4. **观察输出**: 检查MuJoCo仿真窗口和控制台输出

### 常见问题排查
- **导入错误**: 确保在正确的项目根目录下运行
- **文件找不到**: 检查配置文件中的路径是否正确
- **依赖缺失**: 运行 `pip install -r requirements.txt`
- **CUDA问题**: 如果有GPU，可以安装 `onnxruntime-gpu`

### 控制说明
仿真启动后：
- MuJoCo窗口将显示G1机器人执行马步冲拳动作
- 机器人会循环执行动作，到达终点后重置
- 使用MuJoCo的标准控制（鼠标拖拽查看，滚轮缩放等）

### 性能监控
- 观察关节角度是否在限制范围内
- 检查控制频率和仿真稳定性
- 监控策略推理时间
