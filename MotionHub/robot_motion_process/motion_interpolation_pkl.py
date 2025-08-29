import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import joblib
import argparse
from pathlib import Path
import os
# Fix path to be relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
DOF_AXIS_FILE = os.path.join(script_dir, "..", "description", "robots", "g1", "dof_axis.npy")

def lower_dof_interpolation(start_dof,end_dof,nframe):
    mid_frame = nframe // 2
    quat_frame = mid_frame // 2
    left_lower_dof = np.linspace(start_dof[:6], end_dof[:6],
                    num=mid_frame+1,
                    endpoint=False)[1:].reshape(-1, 6)
    right_lower_dof = np.linspace(start_dof[6:12], end_dof[6:12],
                    num=mid_frame+1,
                    endpoint=False)[1:].reshape(-1, 6)
    left_knee_1 = np.linspace(start_dof[3], 1.0,
                    num=quat_frame).reshape(-1, 1)
    left_knee_2 = np.linspace(1.0, end_dof[3],
                    num=quat_frame+1).reshape(-1, 1)
    right_knee_1 = np.linspace(start_dof[9], 1.0,
                    num=quat_frame).reshape(-1, 1)
    right_knee_2 = np.linspace(1.0, end_dof[9],
                    num=quat_frame+1).reshape(-1, 1)
    
    left_lower_dof[:,3] = np.concatenate((left_knee_1, left_knee_2), axis=0).squeeze(-1)
    right_lower_dof[:,3] = np.concatenate((right_knee_1, right_knee_2), axis=0).squeeze(-1)

    left_lower_dof = np.concatenate((left_lower_dof, np.tile(left_lower_dof[-1], (nframe-left_lower_dof.shape[0], 1))), axis=0)
    right_lower_dof = np.concatenate((np.tile(right_lower_dof[0], (nframe-right_lower_dof.shape[0], 1)),right_lower_dof), axis=0)
    
    lower_dof = np.concatenate((left_lower_dof,right_lower_dof), axis=1)
    return lower_dof

def correct_rot(root_rot,root_trans,dof):
    origin_root_rot = root_rot[0]
    q_correction = R.from_quat(origin_root_rot).inv()
    correct_root_rot = (q_correction * R.from_quat(root_rot)).as_quat()
    correct_root_trans = root_trans @ (q_correction.as_matrix().T)
    
    euler_initial = R.from_quat(origin_root_rot).as_euler('xyz',degrees=False)
    theta = euler_initial[1]

    print('correct theta: ',theta)
    
    dof[:,[0,6,14]] += theta

    return correct_root_trans,correct_root_rot,dof


def convert_pkl(data, output_filename,contact_mask):
    data = data.astype(np.float32)
    root_trans = data[:, :3]
    root_qua = data[:, 3:7]
    dof_new = data[:, 7:]
    root_aa = R.from_quat(root_qua).as_rotvec()

    dof_axis = np.load(DOF_AXIS_FILE, allow_pickle=True)
    dof_axis = dof_axis.astype(np.float32)

    pose_aa = np.concatenate(
        (np.expand_dims(root_aa, axis=1), dof_axis * np.expand_dims(dof_new, axis=2), np.zeros((data.shape[0], 3, 3))),
        axis=1).astype(np.float32)

    data_dump = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof_new,
        "root_rot": root_qua,
        "smpl_joints": np.zeros_like(pose_aa),
        "fps": 30,
        "contact_mask":contact_mask
    }

    print("Output DOF Shape",dof_new.shape)
    print("Output Filename: ", output_filename+".pkl")
    all_data = {}
    all_data[output_filename] = data_dump
    joblib.dump(all_data, f'{output_filename}.pkl')


def interpolate_motion(input_data, start_ext_frames, end_ext_frames, default_pose, output_pkl,contact_mask,fix_root_rot, knee_modify):
    root_trans = input_data[:, :3]
    root_rot = input_data[:, 3:7]
    dof_pos = input_data[:, 7:]

    # root_trans,root_rot,dof_pos = correct_rot(root_rot,root_trans,dof_pos)

    start_rot_aa = R.from_quat(root_rot[0]).as_euler('ZYX')
    end_rot_aa = R.from_quat(root_rot[-1]).as_euler('ZYX')
    start_rot_fixed_z = start_rot_aa[0]
    end_rot_fixed_z = end_rot_aa[0]

    default_rt = default_pose[0:3]
    default_rr = default_pose[3:7]
    default_dof = default_pose[7:]
    default_rr_aa = R.from_quat(default_rr).as_euler('ZYX')


    # 起始处插值
    start_rr, start_dof = [], []
    if start_ext_frames > 0:
        # root trans
        start_z = np.linspace(default_rt[2],
                              root_trans[0, 2],
                              start_ext_frames)
        start_root_trans = np.zeros((start_ext_frames, 3))
        start_root_trans[:, 0] = root_trans[0, 0]  # X保持首帧值
        start_root_trans[:, 1] = root_trans[0, 1]  # Y保持首帧值
        start_root_trans[:, 2] = start_z

         # dof pos
        # start_dof = np.linspace(default_dof, dof_pos[0],
        #                         num=start_ext_frames + 1,
        #                         endpoint=False)[1:].reshape(-1, 23)
        if knee_modify:
            upper_start_dof = np.linspace(default_dof[12:], dof_pos[0][12:],
                                    num=start_ext_frames + 1,
                                    endpoint=False)[1:].reshape(-1, 11)
            lower_start_dof = lower_dof_interpolation(default_dof[0:12],dof_pos[0][0:12],start_ext_frames)
            start_dof = np.concatenate((lower_start_dof,upper_start_dof),axis=1)
        else: 
            #  linear interpolation
            start_dof = np.linspace(default_dof, dof_pos[0],
                                    num=start_ext_frames + 1,
                                    endpoint=False)[1:].reshape(-1, 23)

        # root rot
        if not fix_root_rot:
            # breakpoint()
            rotations = R.from_euler('ZYX', 
                                [   np.concatenate((start_rot_aa[0:1],default_rr_aa[1:])), 
                                    np.concatenate((start_rot_aa[0:1],start_rot_aa[1:]))  ])
            # breakpoint()
            times = np.linspace(0, 1, start_ext_frames)
            # slerp 插值
            slerp = Slerp([0, 1], rotations)
            # interp_rots = slerp(times).as_quat()
            interp_rots = slerp(times).as_euler('ZYX')
            # start_rr = np.array([np.array([start_rot_fixed_z,q[1],q[2]]) for q in interp_rots])
            start_rr = R.from_euler('ZYX',interp_rots).as_quat()


    # 结束处插值
    end_rr, end_dof = [], []
    if end_ext_frames > 0:
        # 根位移处理（仅Z轴插值）
        end_z = np.linspace(root_trans[-1, 2],
                            default_rt[2],
                            end_ext_frames)
        end_root_trans = np.zeros((end_ext_frames, 3))
        end_root_trans[:, 0] = root_trans[-1, 0]  # X保持首帧值
        end_root_trans[:, 1] = root_trans[-1, 1]  # Y保持首帧值
        end_root_trans[:, 2] = end_z

        # dof pos
        # end_dof = np.linspace(dof_pos[-1], default_dof,
        #                       num=end_ext_frames + 1)[1:].reshape(-1, 23)
        if knee_modify:
            upper_end_dof = np.linspace(dof_pos[-1][12:], default_dof[12:],
                                    num=end_ext_frames + 1)[1:].reshape(-1, 11)
            lower_end_dof = lower_dof_interpolation(dof_pos[-1][0:12],default_dof[0:12],end_ext_frames)
            end_dof = np.concatenate((lower_end_dof,upper_end_dof),axis=1)
        else:
            end_dof = np.linspace(dof_pos[-1], default_dof,num=end_ext_frames + 1)[1:].reshape(-1, 23)
        
        if not fix_root_rot:
            # root rot
            # end_rotations = R.from_quat([root_rot[-1], default_rr])
            end_rotations = R.from_euler('ZYX', 
                                        [   np.concatenate((end_rot_aa[0:1],default_rr_aa[1:])), 
                                            np.concatenate((end_rot_aa[0:1],end_rot_aa[1:]))  ])
            # times = np.linspace(0, 1, end_ext_frames + 2)[1:-1]
            times = np.linspace(1,0,end_ext_frames)
            slerp = Slerp([0, 1], end_rotations)
            # interp_rots = slerp(times).as_quat()
            interp_rots = slerp(times).as_euler('ZYX')
            # end_rr = np.array([q for q in interp_rots])
            # end_rr = np.array([np.array([end_rot_fixed_z,q[1],q[2]]) for q in interp_rots])
            end_rr = R.from_euler('ZYX',interp_rots).as_quat()

    new_root_trans = np.vstack([
        start_root_trans,
        root_trans,
        end_root_trans
    ])

    if not fix_root_rot:
        new_root_rot = np.vstack([
            start_rr,
            root_rot,
            end_rr
        ])
    else:
        total_frame = start_ext_frames + input_data.shape[0] + end_ext_frames
        new_root_rot = np.tile(default_rr, (total_frame, 1))

    new_dof_pos = np.vstack([
        start_dof,
        dof_pos,
        end_dof
    ])

    # # root trans
    # first_vector = root_trans[0, :].reshape(1, 3)
    # last_vector = root_trans[-1, :].reshape(1, 3)
    # arr_start = np.tile(first_vector, (30, 1))
    # arr_end = np.tile(last_vector, (30, 1))
    # new_root_trans = np.vstack((arr_start, root_trans, arr_end))

    # first_vector = root_rot[0, :].reshape(1, 4)
    # last_vector = root_rot[-1, :].reshape(1, 4)
    # arr_start = np.tile(first_vector, (30, 1))
    # arr_end = np.tile(last_vector, (30, 1))
    # new_root_rot = np.vstack((arr_start, root_rot, arr_end))

    output_data = np.concatenate((new_root_trans, new_root_rot, new_dof_pos), axis=1)

    # np.save('test.npy', output_data)
    convert_pkl(output_data, output_pkl,contact_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_file_name', type=str, help="Origin File name, shape (nframe, 36)",
                        default='dance1_subject2.pkl')
    parser.add_argument('--fix_root_rot', type=bool, help="Fix default pose root rot. A DEBUG option, not a regular usage.",
                        default=False)
    parser.add_argument('--knee_modify', type=bool, help="Modify knee dof when motion interpolation",
                        default=False)
    parser.add_argument('--start', type=int, help="Start frame", default=0)
    parser.add_argument('--end', type=int, help="End frame", default=-1)

    parser.add_argument('--default_pose', type=str, help="Default Pose File name, shape (36,)")
    parser.add_argument('--start_inter_frame', type=int, help="Start Inter frame", default=30)
    parser.add_argument('--end_inter_frame', type=int, help="End Inter frame", default=30)
    args = parser.parse_args()

    origin_file_name = args.origin_file_name
    data = next(iter(joblib.load(f'{args.origin_file_name}').values()))
    
    print("Input DOF Shape: ",data['dof'].shape)
    if args.end ==-1:
        args.end = data['dof'].shape[0]

    print(f"Clip to {(args.start,args.end)}")
    dof = data['dof'][args.start:args.end]
    root_trans = data['root_trans_offset'][args.start:args.end]
    root_rot = data['root_rot'][args.start:args.end]
    contact_mask = data['contact_mask'][args.start:args.end]

    fix_root_rot = args.fix_root_rot
    knee_modify = args.knee_modify
    
    input_data = np.concatenate((root_trans,root_rot,dof), axis=1)

    if args.default_pose is not None:
        default_pose = np.load(args.default_pose)
    else:
        default_pose = np.array([0.0, 0.0, 0.80, 
                                 0.0, 0.0, 0.0, 1.0, 
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                 0.0, 0.0, 0.0, 
                                 0.0, 0.0, 0.0, 0.0, 
                                 0.0, 0.0, 0.0, 0.0])
        default_pose = np.array([0.0, 0.0, 0.80, 
                                 0.0, 0.0, 0.0, 1.0, 
                                 -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                 -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                 0.0, 0.0, 0.0, 
                                 0.2, 0.2, 0.0, 0.9, 
                                 0.2, -0.2, 0.0, 0.9])

    start_ext_frames = args.start_inter_frame
    end_ext_frames = args.end_inter_frame

    if False:
        cit = [1.,1.]
    else:
        cit = [0.5,0.5]
    print("Contact in interpolation: ",cit)

    if knee_modify:
        if start_ext_frames%2==0:
            contact_mask_start = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (start_ext_frames//2-1, 1)),np.tile([1.,0.], (start_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)
            
        else:
            contact_mask_start = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (start_ext_frames//2-1, 1)),np.array([[1.,1.]]),np.tile([1.,0.], (start_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)

        if end_ext_frames%2==0:
            contact_mask_end = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (end_ext_frames//2-1, 1)),np.tile([1.,0.], (end_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)
        else:
            contact_mask_end = np.concatenate((np.array([[1.,1.]]),np.tile([0.,1.], (end_ext_frames//2-1, 1)),np.array([[1.,1.]]),np.tile([1.,0.], (end_ext_frames//2-1, 1)),np.array([[1.,1.]])),axis=0)
    else:
        contact_mask_start = np.tile([cit], (start_ext_frames, 1))
        contact_mask_end = np.tile([cit], (end_ext_frames, 1))
    # contact_mask_start = np.tile([cit], (start_ext_frames, 1))
    # contact_mask_end = np.tile([cit], (end_ext_frames, 1))

    contact_mask = np.concatenate((contact_mask_start, contact_mask, contact_mask_end), axis=0)

    output_filename = str(Path(args.origin_file_name).parent / Path(args.origin_file_name).stem) + f'_inter{cit[0]}_S{args.start}-{start_ext_frames}_E{args.end}-{end_ext_frames}'


    interpolate_motion(
        input_data=input_data,
        start_ext_frames=start_ext_frames,
        end_ext_frames=end_ext_frames,
        default_pose=default_pose,
        output_pkl=output_filename,
        contact_mask=contact_mask,
        fix_root_rot=fix_root_rot,
        knee_modify = knee_modify
    )