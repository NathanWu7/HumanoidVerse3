import bpy
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot
import numpy as np

npz_path = "Path_TO_NPZ"
body_model_type = "SMPL-male"
gender = "neutral"
fps = 30 

try:
    data = np.load(npz_path, allow_pickle=True)
    print("Loaded keys:", list(data.keys()))
except Exception as e:
    raise Exception(f"can not import the file: {str(e)}")

poses = data['poses'][:, :72]
betas = data['betas']
trans = data['trans']

if body_model_type not in bpy.data.objects:
    raise Exception(f"please first import {body_model_type} to scence")

body = bpy.data.objects[body_model_type]

if 'betas' in data:
    if hasattr(body, 'shape_keys'):
        for i in range(10):
            key = body.shape_keys.key_blocks.get(f'Shape{i:03d}')
            if key:
                key.value = betas[i]

bone_list = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle',
             'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar',
             'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
             'R_Wrist', 'L_Hand', 'R_Hand']

bpy.context.scene.frame_end = len(poses)
for frame_idx in range(len(poses)):
    pose = poses[frame_idx]
    trans_vec = trans[frame_idx]

    i = 0
    for bone in bone_list:
        bone = body.pose.bones[bone]
        bone.rotation_mode = 'XYZ'
        if bone=='Pelvis':
            bone.rotation_euler = (sRot.from_rotvec(pose[:3]) * sRot.from_euler('xyz', np.array([np.pi / 2, 0, np.pi]))).as_rotvec()
        else:
            bone.rotation_euler = (pose[i*3], pose[i*3+1], pose[i*3+2])
        bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
        i+=1

    root_bone = body.pose.bones['Pelvis']
    root_bone.location = (trans_vec[0], trans_vec[1], trans_vec[2])
    root_bone.keyframe_insert(data_path="location", frame=frame_idx)
    #root_bone.location = trans_vec

    body.keyframe_insert(data_path="location", frame=frame_idx)
    body.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

    print(f"Processed frame {frame_idx}/{len(poses)}")

bpy.context.scene.render.fps = fps
bpy.context.scene.frame_preview_start = 0
bpy.context.scene.frame_preview_end = len(poses)

print("Import Success！")
