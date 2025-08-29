
### Hardware Requirements

We test the code in the following environment:
- **OS**: Ubuntu 20.04
- **GPU**: NVIDIA RTX 4090, Driver Version: 560.35.03 
- **CPU**: 13th Gen Intel(R) Core(TM) i7-13700



### Environment Setup
```bash
# Assuming pwd: PBHC/
conda create -n pbhc python=3.8
conda activate pbhc

# Install and Test IsaacGym
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
pip install -e isaacgym/python
cd isaacgym/python/examples
python 1080_balls_of_solitude.py # or `python joint_monkey.py`
cd ../../..


# Install PBHC
pip install -e .
pip install -e humanoidverse/isaac_utils

```

```bash
# (Optional) Install additional dependencies for motion visualization with rerun (robot_motion_process/vis_rr.py)
pip install rerun-sdk==0.22.0 trimesh
```

