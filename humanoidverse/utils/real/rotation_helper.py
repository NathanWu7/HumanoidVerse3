import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    # WXYZ
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    # imu_quat: WXYZ
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def quaternion_to_euler_array(quat:np.ndarray)->np.ndarray:
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])



def rpy_to_quaternion_array(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw (radians) to quaternion [x, y, z, w]"""
    roll, pitch, yaw = rpy

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])



def quat_mul_np(q1:np.ndarray, q2:np.ndarray)->np.ndarray:
    # q1: XYZW # q2: XYZW
    # q1: XYZW # q2: XYZW
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([x, y, z, w])  # XYZW

def random_perturb_quaternion(quat:np.ndarray, max_angle:float)->np.ndarray:
    """
    Randomly perturb a quaternion by a random axis and angle
    """
    # quat: XYZW
    
    # Generate random axis and angle for perturbation
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    angle = max_angle * np.random.rand()  # Random angle within max_angle
    
    # Convert angle-axis to quaternion
    sin_half_angle = np.sin(angle / 2)
    cos_half_angle = np.cos(angle / 2)
    perturb_quat = np.array([sin_half_angle * axis[0],
                            sin_half_angle * axis[1], 
                            sin_half_angle * axis[2],
                            cos_half_angle])  # XYZW format
    
    # Multiply quaternions to apply perturbation
    return quat_mul_np(perturb_quat, quat)