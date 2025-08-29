import argparse
import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh
import joblib
import math

class RerunURDF():
    def __init__(self, robot_type):
        self.name = robot_type
        self.robot = pin.RobotWrapper.BuildFromURDF('description/robots/g1/29dof_official.urdf', 'description/robots/g1', pin.JointModelFreeFlyer())
        self.Tpose = np.array([0,0,0.785,0,0,0,1,
                                -0.15,0,0,0.3,-0.15,0,
                                -0.15,0,0,0.3,-0.15,0,
                                0,0,0,
                                0, 1.57,0,1.57,0,0,0,
                                0,-1.57,0,1.57,0,0,0]).astype(np.float32)
        
        # print all joints names
        # for i in range(self.robot.model.njoints):
        #     print(self.robot.model.names[i])
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            # print(parent_joint_name)
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))


def quat_to_euler(x, y, z, w):
    """Convert quaternion (xyzw) to Euler angles (roll, pitch, yaw) in radians"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        print("Warning: Pitch out of range, using 90 degrees")
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles (radians) to quaternion (xyzw)"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.array([x, y, z, w])

def remove_yaw(quaternions):
    """Remove yaw component by converting to Euler angles and zeroing yaw"""
    eulers = np.array([quat_to_euler(*q) for q in quaternions])
    eulers[:, 2] = 0  # Zero yaw
    new_quats = np.array([euler_to_quat(r, p, y) for r, p, y in eulers])
    return new_quats

def rebase_yaw(quaternions):
    """Remove yaw component by converting to Euler angles and zeroing yaw"""
    eulers = np.array([quat_to_euler(*q) for q in quaternions])
    # eulers[:, 2] = 0  # Zero yaw
    eulers[:, 2] -= eulers[0, 2] 
    new_quats = np.array([euler_to_quat(r, p, y) for r, p, y in eulers])
    return new_quats



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="File path",required=True)
    args = parser.parse_args()

    rr.init('Reviz', spawn=True)
    # rr.init(args.filepath, spawn=True)
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    filepath = args.filepath
    robot_type = 'g1'
    data = joblib.load(filepath)
    keyname = list(data.keys())[0]
    data = data[keyname]
    dof = data['dof']
    L_Wrist_dof = np.zeros((dof.shape[0],3))
    R_Wrist_dof = np.zeros((dof.shape[0],3))
    motion_data = np.concatenate((data['root_trans_offset'],data['root_rot'],dof[:,0:19],L_Wrist_dof,dof[:,19:],R_Wrist_dof),axis=1)
    # breakpoint()
    motion_data[:,:3]*=0
    # motion_data[:,3:7] = remove_yaw(motion_data[:,3:7])
    motion_data[:,3:7] = rebase_yaw(motion_data[:,3:7])
    

    rerun_urdf = RerunURDF(robot_type)
    for frame_nr in range(motion_data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        configuration = motion_data[frame_nr, :]
        rerun_urdf.update(configuration)
