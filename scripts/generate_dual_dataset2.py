import os
import numpy as np
from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, GraspableObject3D, GraspQualityConfigFactory, GraspQualityFunctionFactory, RobotGripper, PointGraspMetrics3D
from autolab_core import YamlConfig
from meshpy import ObjFile, SdfFile
import h5py
from dexnet.api import DexNet
import multiprocessing as mp


config_filename ="../api_config.yaml"
file_dir = "/home/ymxlzgy/Downloads/dataset/simplified"
grasp_dir = "../grasp/"

# turn relative paths absolute
if not os.path.isabs(config_filename):
    config_filename = os.path.join(os.getcwd(), config_filename)

config = YamlConfig(config_filename)
gripper_name = "robotiq_85"
params={'friction_coef': config['sampling_friction_coef']}

def mesh_antipodal_grasp_sampler(file_name):
    of = ObjFile(file_name)

    mesh = of.read()

    obj = GraspableObject3D(None, mesh)

    gripper = RobotGripper.load(gripper_name, "./grippers")

    scale, f, d, t, G, grasps = DexNet._single_obj_grasps(None, obj, gripper, config, stable_pose_id=None)

    return scale, f, d, t, grasps, gripper

def generate_pose(num, file):
    OBJ_FILENAME = file
    print("Deal with {}".format(OBJ_FILENAME), "Process {}".format(str(num)))
    scale, f, d, t, g, gripper = mesh_antipodal_grasp_sampler(file)
    g=np.array(g)
    data = h5py.File(grasp_dir + OBJ_FILENAME.split('.')[0].split('/')[-1] + '_{}'.format(scale) + '.h5', 'w')
    temp1 = data.create_group("grasps")
    temp1["axis"] = [(g[i][0].axis, g[i][1].axis) for i in range(len(g))]
    temp1["angle"] = [(g[i][0].approach_angle, g[i][1].approach_angle) for i in range(len(g))]
    temp1["center"] = [(g[i][0].center, g[i][1].center) for i in range(len(g))]
    temp1["end_points"] = [(g[i][0].endpoints, g[i][1].endpoints) for i in range(len(g))]
    temp1["grasp_points"] = [(g[i][0].grasp_point1, g[i][0].grasp_point2, g[i][1].grasp_point1, g[i][1].grasp_point2) for i in range(len(g))]
    temp1["transforms"] = [((g[i][0].gripper_pose(gripper) * gripper.T_mesh_gripper.inverse()).matrix, (g[i][1].gripper_pose(gripper) * gripper.T_mesh_gripper.inverse()).matrix) for i in range(len(g))]
    temp1["qualities/Force_closure"] = np.array(f).reshape(len(f), -1) if d.size else []
    temp1["qualities/Dexterity"] = np.array(d).reshape(len(d), -1) if d.size else []
    temp1["qualities/Torque_optimization"] = np.array(t).reshape(len(t), -1) if t.size else []
    temp2 = data.create_group("object")
    temp2["file"] = OBJ_FILENAME.split('/')[-1]
    temp2["scale"] = scale
    print("Done for {}".format(OBJ_FILENAME), "Process {}".format(str(num)))

def print_error(value):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ")
    print(value)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ")

def main():
    pool = mp.Pool(3)
    file_list = []
    existed_list = []
    for root, dirs, files in os.walk(file_dir):
        file_list = [root + '/' + file for file in files]
        file_list.sort()

    for root, dirs, files in os.walk(grasp_dir):
        existed_list = [file.split('_')[0] for file in files]
        existed_list.sort()

    for i in range(len(file_list)*3//4, len(file_list)):
        if file_list[i].split('/')[-1].split('.')[0] in existed_list:
            print("skip the {}".format(file_list[i]))
            continue
        pool.apply_async(generate_pose, args=(i, file_list[i]),
                         error_callback=print_error)
    pool.close()
    pool.join()
    print("Done!")

if __name__ == "__main__":
    main()
