# HumanoidVerse3 
本项目基于 [HumanoidVerse ](README_V1.md) 和 [HumanoidVerse2](README_V2.md) 开发，环境配置与初始操作请参考 [HumanoidVerse ](README_HumanoidVerse.md)。

旨在解决不同版本和依赖问题，进行一些debug，集成 sim2real 部分，最终补充部分动作序列转换方法与数据。

[HumanoidVerse3 English](README_V3.md)

# 训练
请参考训练指令文档。

[指令文档](Commands.md)

# 仿真到现实（Sim2real）
训练完成后，请分别在基于 Isaac 和 Mujoco 系列的模拟器（如 Isaac Gym、Isaac Sim、Genesis 等）中进行测试。如果性能达到预期且基本一致，可进行 sim2real 部署测试。

本仓库包含了一个简化版的 sim2real 仓库 **Humanoid_robot_deployment**。原始版本请参考
[Humanoid_robot_deployment](https://github.com/YixFeng/Humanoid_robot_deployment)

指令请参考指令文档。
[指令文档](Commands.md)

# 运动数据
如需从视频中提取运动并进行数据集转换，请参考 [PBHC](https://github.com/TeleHuman/PBHC)。本仓库后续会更新部分转换脚本。

# 许可证

本项目基于 MIT 许可证开源，详情请参见 [LICENSE](LICENSE) 文件。
