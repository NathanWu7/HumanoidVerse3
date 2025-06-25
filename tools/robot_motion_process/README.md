
# Robot Motion Processing

In this folder, we provide tools for visualizing and analyzing the processed motion and the motion trajectory from simulation.

All of the retargeted reference motion and motion trajectory collected from simulation take the same data format. `motion_readpkl.py` provide an example code to load them.

To collect the motion trajectory in IsaacGym, use `humanoidverse/sample_eps.py`. For MuJoCo, use `humanoidverse/urci.py`, the program will automatically save the motion trajectory to a pickle file when each episode ends.



## Visualization

`vis_q_mj.py` and `vis_rr.py` are both used to visualize the motion data. These two scripts load the same data format, but have a different GUI. `vis_q_mj.py` is based on `mujoco_py`, provide reference motion & torque visualization and support manually correct the contact mask. `vis_rr.py` is based on `rerun`, provide more interactive visualization.



Usage:

```bash
python robot_motion_process/vis_q_mj.py +motion_file=...
```
See `def key_call_back(keycode)` in `vis_q_mj.py` for the key mapping.

```bash
python robot_motion_process/vis_rr.py --filepath=...
```

Note that you need to install the additional dependencies for `vis_rr.py`.

```bash
# (Optional) Install additional dependencies for visualization
pip install rerun-sdk==0.22.0 trimesh
```



## Interpolation

`motion_interpolation_pkl.py` is used to preprocess the motion data. Given a default robot pose, this script will add a linear interpolation between the default pose and the motion frames at both the beginning and the end. So that the processed motion data all begin and end with the default pose. 

This is useful for deployment since it's inconvenient to initialize the real-world robot to the beginning pose of each motion. Some pose may be strange so that the robot cannot stand stably.

Usage:
```bash
python robot_motion_process/motion_interpolation_pkl.py +origin_file_name=... +start=... +end=... +start_inter_frame=... +end_inter_frame=...
```
- `origin_file_name` is the path to the original motion data, it should be a pickle file.
- `start` and `end` are the start and end frame index of the motion data, we select the `[start:end]` frames from the original motion data.
- `start_inter_frame` and `end_inter_frame` are the number of the added interpolation frames at the beginning and the end of the motion data.


## Trajectory Analysis

`traj_vis.ipynb` is used to visualize the motion trajectory. For motion data collected from policy rollout, you can use it to compare the motion trajectory with the reference motion. 

Usage: Change the path to motion data in the notebook according to the example cells and run all cells.



