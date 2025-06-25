# HumanoidVerse3 
本项目基于 [HumanoidVerse ](README_V1.md) 和 [HumanoidVerse2](README_V2.md) 开发，环境配置与初始操作请参考 [HumanoidVerse ](README_HumanoidVerse.md)。

旨在解决不同版本和依赖问题，进行一些debug，集成 sim2real 部分，最终补充部分动作序列转换方法与数据。

[HumanoidVerse3 English](README_V3.md)

# 在开始之前
不同的仿真环境甚至是真实机器人环境之间是隔离的，这也意味着你可以仅使用某一仿真环境，或直接在真实机器人上进行模型的测试。在使用时，你需要用虚拟环境管理工具（如conda进行管理）。在切换simulator之前，需要激活对应的仿真环境。

对于每一个虚拟环境，请安装主工程依赖和索引
```bash
conda activate each_of_your_env

pip install -e .
```

其他训练命令
```bash
conda activate yourenv

other commands

conda activate/deactivate
```

# 训练
请参考训练指令文档。

[指令文档](Commands.md)

# 仿真到现实（Sim2real）
训练完成后，请分别在基于 Isaac 和 Mujoco 系列的模拟器（如 Isaac Gym、Isaac Sim、Genesis 等）中进行测试。如果性能达到预期且基本一致，可进行 sim2real 部署测试。

本仓库包含了一个简化版的 sim2real 仓库 **humanoid_sim2real**。原始版本请参考
[Humanoid_robot_deployment](https://github.com/YixFeng/Humanoid_robot_deployment)

指令请参考指令文档。
[指令文档](Commands.md)

# 运动数据
如需从视频中提取运动并进行数据集转换，请参考 [PBHC](https://github.com/TeleHuman/PBHC)。本仓库后续会更新部分转换脚本。

# 参考
感谢以下论文和代码的作者

```bibtex
@misc{HumanoidVerse2,
  author = {liangjun},
  title = {HumanoidVerse2: A Multi-Simulator Framework with Modular Design for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zengliangjun/HumanoidVerse2}},
}

@misc{HumanoidVerse,
  author = {CMU LeCAR Lab},
  title = {HumanoidVerse: A Multi-Simulator Framework for Humanoid Robot Sim-to-Real Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LeCAR-Lab/HumanoidVerse}},
}

@article{xie2025kungfubot,
  title={KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills},
  author={Xie, Weiji and Han, Jinrui and Zheng, Jiakun and Li, Huanyu and Liu, Xinzhe and Shi, Jiyuan and Zhang, Weinan and Bai, Chenjia and Li, Xuelong},
  journal={arXiv preprint arXiv:2506.12851},
  year={2025}
}

@misc{Humanoid_robot_deployment,
  author = {Yixiao Feng, Yuetong Fang},
  title = {Humanoid_robot_deployment},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YixFeng/Humanoid_robot_deployment}},
}

```
# 引用
如果引用本仓库
```bibtex
@misc{HumanoidVerse3,
  author = {Qiwei Wu},
  title = {HumanoidVerse3},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nathanwu7/HumanoidVerse3}},
}
```

# 许可证

本项目基于 MIT 许可证开源，详情请参见 [LICENSE](LICENSE) 文件。
