import numpy as np
from scipy.spatial.transform import Rotation as sRot
import joblib
import argparse
import os


def convert(file_name, data):
    data = data.astype(np.float32)
    print(data.shape)
    root_trans = data[:, :3]
    root_qua = data[:, 3:7]
    dof = data[:, 7:]

    dof_new = np.concatenate((dof[:, :19], dof[:, 22:26]), axis=1)
    root_aa = sRot.from_quat(root_qua).as_rotvec()

    dof_axis = np.load('../description/robots/g1/dof_axis.npy', allow_pickle=True)
    dof_axis = dof_axis.astype(np.float32)

    pose_aa = np.concatenate(
        (np.expand_dims(root_aa, axis=1), dof_axis * np.expand_dims(dof_new, axis=2), np.zeros((data.shape[0], 3, 3))),
        axis=1).astype(np.float32)

    data_dump = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": dof_new,
        "root_rot": root_qua,
        "smpl_joints": pose_aa,
        "fps": 30
    }

    all_data = {}
    all_data[file_name] = data_dump

    os.makedirs('./lafan_pkl', exist_ok=True)
    joblib.dump(all_data, f'./lafan_pkl/{file_name}.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="File name", required=True)
    parser.add_argument('--start', type=int, help="Start frame", default=0)
    parser.add_argument('--end', type=int, help="End frame", default=100)
    args = parser.parse_args()

    start_frame = args.start
    end_frame = args.end
    robot_type = 'g1'

    filepath = args.filepath
    csv_file = filepath + '.csv'
    data = np.genfromtxt(csv_file, delimiter=',')[start_frame:end_frame, :]

    output_name = filepath.split('/')[-1] + '_' + f'{start_frame}' + '_' + f'{end_frame}'

    convert(output_name, data)