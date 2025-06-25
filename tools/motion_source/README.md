# Motion Source

This folder describes how we collect human motion from various sources and process them to the SMPL format.

## SMPL Format

SMPL format data consists of a dictionary. In details:
```
'betas': (10,) - this is the shape paramater of SMPL model, not important
'gender': str - the gender of SMPL model, default is neutral
'poses': (frames, 66) - joint rotations, 66 represents 22 joints with 3 angles of rotation each.
'trans': (frames, 3) - root translation (global translation)
'mocap_framerate': int - motion fps, amass and video data is 30, mocap data is 120.
```


## AMASS

AMASS motions do not need further processing in this step.

## Video

We use [GVHMR](https://github.com/zju3dv/GVHMR) to extract motions from videos.

## LAFAN (Unitree)

> Note that the Unitree Retargeted `LAFAN` dataset is not available on the public. It's originally released in [here](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset).

The format of **Unitree LAFAN** dataset is `csv`. Two steps to process these data:

1. convert csv to pkl:
```
python convert_lafan_pkl.py --filepath <path_to_csv> --start 0 --end 100
```
- `<path_to_csv> ` is the path of csv file and does not contain `.csv`.
- `start` is the start frame index and `end` is the end frame index. We select the `[start:end]` frames from the csv file.

2. count pkl contact mask:
 
```
python count_pkl_contact_mask.py robot=unitree_g1_29dof_anneal_23dof +input_folder=<path_to_input_folder> +target_folder=<path_to_target_folder>
```
- It computes the contact mask of the motion data by a thresholding method.