import os
import numpy as np
from dexnet.grasping import GraspableObject3D, RobotGripper
from autolab_core import YamlConfig
from meshpy import ObjFile
import h5py
from dexnet.api import DexNet


config_filename ="../api_config.yaml"
OBJ_FILENAME = "/home/ymxlzgy/Downloads/dataset/simplified/1bbe463ba96415aff1783a44a88d6274.obj"


# turn relative paths absolute
if not os.path.isabs(config_filename):
    config_filename = os.path.join(os.getcwd(), config_filename)

config = YamlConfig(config_filename)
gripper_name = "robotiq_85"
final_grasps = []
params={'friction_coef': config['sampling_friction_coef']}
gripper = RobotGripper.load(gripper_name, "./grippers")
def mesh_antipodal_grasp_sampler():
    of = ObjFile(OBJ_FILENAME)

    mesh = of.read()

    obj = GraspableObject3D(None, mesh)



    scale, f, d, t, grasps = DexNet._single_obj_grasps(None, obj, gripper, config, stable_pose_id=None)

    return scale, f, d, t, grasps, gripper


scale, f, d, t, g, gripper = mesh_antipodal_grasp_sampler()
g=np.array(g)
data = h5py.File("../test_tmp/" + OBJ_FILENAME.split('.')[0].split('/')[-1] + '_{}'.format(scale) + '.h5', 'w')
temp1 = data.create_group("grasps")
temp1["axis"] = [(g[i][0].axis, g[i][1].axis) for i in range(len(g))]
temp1["angle"] = [(g[i][0].approach_angle, g[i][1].approach_angle) for i in range(len(g))]
temp1["center"] = [(g[i][0].center, g[i][1].center) for i in range(len(g))]
temp1["end_points"] = [(g[i][0].endpoints, g[i][1].endpoints) for i in range(len(g))]
temp1["grasp_points"] = [(g[i][0].grasp_point1, g[i][0].grasp_point2, g[i][1].grasp_point1, g[i][1].grasp_point2) for i in range(len(g))]
temp1["transforms"] = [((g[i][0].gripper_pose(gripper) * gripper.T_mesh_gripper.inverse()).matrix, (g[i][1].gripper_pose(gripper) * gripper.T_mesh_gripper.inverse()).matrix) for i in range(len(g))]
temp1["qualities/Force_closure"] = np.array(f).reshape(len(f), -1) if f.size else []
temp1["qualities/Dexterity"] = np.array(d).reshape(len(d), -1) if d.size else []
temp1["qualities/Torque_optimization"] = np.array(t).reshape(len(t), -1) if t.size else []
temp2 = data.create_group("object")
temp2["file"] = OBJ_FILENAME.split('/')[-1]
temp2["scale"] = scale
