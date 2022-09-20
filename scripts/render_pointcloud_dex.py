"""
Original: Contact-GraspNet
Modified by: Guangyao Zhai

"""

import glob
import os
import random
import trimesh
import numpy as np
import json
import argparse
import signal
import h5py
import pyrender
import copy
import yaml
import provider
import cv2
import trimesh.transformations as tra
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

from DA2_tools import Scene, load_mesh, create_robotiq_marker
from autolab_core import RigidTransform
from scipy.spatial import cKDTree
from mesh_utils import Object

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vis_points(pc):
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )
    axes = np.array(
        [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0]],
        dtype=np.float64,
    )
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], scale_factor=0.01, color=(1,0,0), figure=fig)
    mlab.show()

def remove_plane(data, distance_threshold=0.05, max_iterations=10):
    '''
    Given a point cloud, return a mask for inliers and the plane model. This method iteractively
    fits plane to inliers points until the set of inliers are unchanged
    Input:
      data: float32 2D (Nx3) numpy array of the input 3D point cloud
      distance_threshold: threshold for point to plane distance for inliers
    Output:
      inliers: boolean numpy array of size N, mask for inliers
      model: plane model of (normal, offset)
    '''
    new_inliers = np.ones((data.shape[0]), dtype=bool)
    for i in range(max_iterations):
        inliers = new_inliers
        if np.sum(inliers) < 3:
            break
        normal, offset = fit_plane_to_points(data[inliers, :])
        dist = np.abs(np.matmul(data, normal) + offset)
        new_inliers = dist <= distance_threshold
        if np.all(inliers == new_inliers):
            break
    return data[~inliers, :], normal

def fit_plane_to_points(data):
    '''
    Fits a plane normal to a set of 3d points using SVD, and return the normal and offset of the
    plane model.
    Input:
      data: float32 2D (Nx3) numpy array of the input 3D point cloud
    Output:
      normal, offset of the plane model
    '''
    centroid = np.mean(data, axis=0)
    X = data - centroid
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    normal = vh[-1, :]
    offset = -np.dot(normal, centroid)
    return normal, offset

def load_splits(root_folder):
    """
    Load splits of training and test objects

    Arguments:
        root_folder {str} -- path to acronym data

    Returns:
        [dict] -- dict of category-wise train/test object grasp files
    """
    split_dict = {}
    split_paths = glob.glob(os.path.join(root_folder, 'splits/*.json'))
    for split_p in split_paths:
        category = os.path.basename(split_p).split('.json')[0]
        splits = json.load(open(split_p,'r'))
        split_dict[category] = {}
        split_dict[category]['train'] = [obj_p.replace('.json', '.h5') for obj_p in splits['train']]
        split_dict[category]['test'] = [obj_p.replace('.json', '.h5') for obj_p in splits['test']]
    return split_dict

def load_grasp_path(root_folder):
    """
    Load splits of training and test objects

    Arguments:
        root_folder {str} -- path to acronym data

    Returns:
        [dict] -- dict of category-wise train/test object grasp files
    """
    split_dict = {}
    split_paths = glob.glob(os.path.join(root_folder, 'grasp/*.h5')) #need to change this name
    return split_paths


def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters

def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates

      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """

    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        print("points too few need compensate!")
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def vectorized_normal_computation(pc, neighbors):
    """
    Vectorized normal computation with numpy

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        neighbors {np.ndarray} -- Nxkx3 neigbours

    Returns:
        [np.ndarray] -- Nx3 normal directions
    """
    diffs = neighbors - np.expand_dims(pc, 1) # num_point x k x 3
    covs = np.matmul(np.transpose(diffs, (0, 2, 1)), diffs) # num_point x 3 x 3
    covs /= diffs.shape[1]**2
    # takes most time: 6-7ms
    eigen_values, eigen_vectors = np.linalg.eig(covs) # num_point x 3, num_point x 3 x 3
    orders = np.argsort(-eigen_values, axis=1) # num_point x 3
    orders_third = orders[:,2] # num_point
    directions = eigen_vectors[np.arange(pc.shape[0]),:,orders_third]  # num_point x 3
    dots = np.sum(directions * pc, axis=1) # num_point
    directions[dots >= 0] = -directions[dots >= 0]
    return directions

def estimate_normals_cam_from_pc(self, pc_cam, max_radius=0.05, k=12):
    """
    Estimates normals in camera coords from given point cloud.

    Arguments:
        pc_cam {np.ndarray} -- Nx3 point cloud in camera coordinates

    Keyword Arguments:
        max_radius {float} -- maximum radius for normal computation (default: {0.05})
        k {int} -- Number of neighbors for normal computation (default: {12})

    Returns:
        [np.ndarray] -- Nx3 point cloud normals
    """
    tree = cKDTree(pc_cam, leafsize=pc_cam.shape[0]+1)
    _, ndx = tree.query(pc_cam, k=k, distance_upper_bound=max_radius, n_jobs=-1) # num_points x k

    for c,idcs in enumerate(ndx):
        idcs[idcs==pc_cam.shape[0]] = c
        ndx[c,:] = idcs
    neighbors = np.array([pc_cam[ndx[:,n],:] for n in range(k)]).transpose((1,0,2))
    pc_normals = vectorized_normal_computation(pc_cam, neighbors)
    return pc_normals

def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.

    Arguments:
        trans {np.ndarray} -- 4x4 transform.

    Returns:
        [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output

def center_pc_convert_cam(cam_poses, batch_data):
    """
    Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose

    :param cam_poses: (bx4x4) Camera poses in OpenGL format
    :param batch_data: (bxNx3) point clouds
    :returns: (cam_poses, batch_data) converted
    """
    # OpenCV OpenGL conversion
    cam_poses_inv = copy.deepcopy(cam_poses)
    for j in range(len(cam_poses)):
        cam_poses[j,:3,1] = -cam_poses[j,:3,1]
        cam_poses[j,:3,2] = -cam_poses[j,:3,2]
        cam_poses_inv[j] = inverse_transform(cam_poses[j])


    return cam_poses, cam_poses_inv, batch_data

class SceneRenderer:
    def __init__(self, intrinsics=None, fov=np.pi / 6, caching=True, viewing_mode=False):
        """Renders depth with given intrinsics during training.

        Keyword Arguments:
            intrinsics {str} -- camera name from 'kinect_azure', 'realsense' (default: {None})
            fov {float} -- field of view, ignored if inrinsics is not None (default: {np.pi/6})
            caching {bool} -- whether to cache object meshes (default: {True})
            viewing_mode {bool} -- visualize scene (default: {False})
        """

        self._fov = fov

        self._scene = pyrender.Scene()
        self._table_dims = [4.0, 4.8, 0.6]
        self._table_pose = np.eye(4)
        self._viewer = viewing_mode
        if viewing_mode:
            self._viewer = pyrender.Viewer(
                self._scene,
                use_raymond_lighting=True,
                run_in_thread=True)

        self._intrinsics = intrinsics
        if self._intrinsics == 'realsense':
            self._fx=616.36529541
            self._fy=616.20294189
            self._cx=310.25881958
            self._cy=236.59980774
            self._znear=0.04
            self._zfar=20
            self._height=480
            self._width=640
        elif self._intrinsics == 'kinect_azure':
            self._fx=631.54864502
            self._fy=631.20751953
            self._cx=638.43517329
            self._cy=366.49904066
            self._znear=0.04
            self._zfar=20
            self._height=720
            self._width=1280

        self._add_table_node()
        self._init_camera_renderer()

        self._current_context = None
        self._cache = {} if caching else None
        self._caching = caching

    def _init_camera_renderer(self):
        """
        If not in visualizing mode, initialize camera with given intrinsics
        """

        if self._viewer:
            return

        if self._intrinsics in ['kinect_azure', 'realsense']:
            camera = pyrender.IntrinsicsCamera(self._fx, self._fy, self._cx, self._cy, self._znear, self._zfar)
            self._camera_node = self._scene.add(camera, pose=np.eye(4), name='camera')
            self.renderer = pyrender.OffscreenRenderer(viewport_width=self._width,
                                                       viewport_height=self._height,
                                                       point_size=1.0)
        else:
            camera = pyrender.PerspectiveCamera(yfov=self._fov, aspectRatio=1.0, znear=0.001) # do not change aspect ratio
            self._camera_node = self._scene.add(camera, pose=tra.euler_matrix(np.pi, 0, 0), name='camera')
            self.renderer = pyrender.OffscreenRenderer(400, 400)


    def get_trimesh_camera(self):
        """Get a trimesh object representing the camera intrinsics.

        Returns:
            trimesh.scene.cameras.Camera: Intrinsic parameters of the camera model
        """
        return trimesh.scene.cameras.Camera(
            focal=(self._fx, self._fy),
            resolution=(self._width, self._height),
            z_near=self._znear,
        )
    def _add_table_node(self):
        """
        Adds table mesh and sets pose
        """
        if self._viewer:
            return
        table_mesh = trimesh.creation.box(self._table_dims)
        mesh = pyrender.Mesh.from_trimesh(table_mesh)

        table_node = pyrender.Node(mesh=mesh, name='table')
        self._scene.add_node(table_node)
        self._scene.set_pose(table_node, self._table_pose)


    def _load_object(self, path, scale):
        """
        Load a mesh, scale and center it

        Arguments:
            path {str} -- path to mesh
            scale {float} -- scale of the mesh

        Returns:
            dict -- contex with loaded mesh info
        """
        if (path, scale) in self._cache:
            return self._cache[(path, scale)]
        obj = Object(path, scale)

        tmesh = obj.mesh
        tmesh_mean = np.mean(tmesh.vertices, 0) # TODO
        # tmesh.vertices -= np.expand_dims(tmesh_mean, 0)

        lbs = np.min(tmesh.vertices, 0)
        ubs = np.max(tmesh.vertices, 0)
        object_distance = np.max(ubs - lbs) * 5

        mesh = pyrender.Mesh.from_trimesh(tmesh)

        context = {
            'name': path + '_' + str(scale),
            'tmesh': copy.deepcopy(tmesh),
            'distance': object_distance,
            'node': pyrender.Node(mesh=mesh, name=path + '_' + str(scale)),
            'mesh_mean': np.expand_dims(tmesh_mean, 0),
        }

        self._cache[(path, scale)] = context

        return self._cache[(path, scale)]


    def change_scene(self, obj_paths, obj_scales, obj_transforms):
        """Remove current objects and add new ones to the scene

        Arguments:
            obj_paths {list} -- list of object mesh paths
            obj_scales {list} -- list of object scales
            obj_transforms {list} -- list of object transforms
        """
        if self._viewer:
            self._viewer.render_lock.acquire()
        for n in self._scene.get_nodes():
            if n.name not in ['table', 'camera', 'parent']:
                self._scene.remove_node(n)

        if not self._caching:
            self._cache = {}

        for p,t,s in zip(obj_paths, obj_transforms, obj_scales):

            object_context = self._load_object(p, s)
            object_context = copy.deepcopy(object_context)

            self._scene.add_node(object_context['node'])
            self._scene.set_pose(object_context['node'], t)

        if self._viewer:
            self._viewer.render_lock.release()

    def _to_pointcloud(self, depth):
        """Convert depth map to point cloud

        Arguments:
            depth {np.ndarray} -- HxW depth map

        Returns:
            np.ndarray -- Nx4 homog. point cloud
        """
        if self._intrinsics in ['kinect_azure', 'realsense']:
            fy = self._fy
            fx = self._fx
            height = self._height
            width = self._width
            cx = self._cx
            cy = self._cy

            mask = np.where(depth > 0)

            x = mask[1]
            y = mask[0]

            normalized_x = (x.astype(np.float32) - cx)
            normalized_y = (y.astype(np.float32) - cy)
        else:
            fy = fx = 0.5 / np.tan(self._fov * 0.5) # aspectRatio is one.
            height = depth.shape[0]
            width = depth.shape[1]

            mask = np.where(depth > 0)

            x = mask[1]
            y = mask[0]

            normalized_x = (x.astype(np.float32) - width * 0.5) / width
            normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T


    def render(self, pose, render_pc=True):
        """Render object or scene in camera pose

        Arguments:
            pose {np.ndarray} -- 4x4 camera pose

        Keyword Arguments:
            render_pc {bool} -- whether to convert depth map to point cloud (default: {True})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- HxWx3 color, HxW depth, Nx4 point cloud, 4x4 camera pose
        """

        transferred_pose = pose.copy()
        self._scene.set_pose(self._camera_node, transferred_pose)

        color, depth = self.renderer.render(self._scene)

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, transferred_pose

    def render1(self, pose, render_pc=True):
        """Render object or scene in camera pose

        Arguments:
            pose {np.ndarray} -- 4x4 camera pose

        Keyword Arguments:
            render_pc {bool} -- whether to convert depth map to point cloud (default: {True})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- HxWx3 color, HxW depth, Nx4 point cloud, 4x4 camera pose
        """

        transferred_pose = pose.copy()
        self._scene.set_pose(self._camera_node1, transferred_pose)

        color, depth = self.renderer.render(self._scene)

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, transferred_pose

    def render2(self, pose, render_pc=True):
        """Render object or scene in camera pose

        Arguments:
            pose {np.ndarray} -- 4x4 camera pose

        Keyword Arguments:
            render_pc {bool} -- whether to convert depth map to point cloud (default: {True})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- HxWx3 color, HxW depth, Nx4 point cloud, 4x4 camera pose
        """

        transferred_pose = pose.copy()
        self._scene.set_pose(self._camera_node2, transferred_pose)

        color, depth = self.renderer.render(self._scene)

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None

        return color, depth, pc, transferred_pose

    def render_labels(self, full_depth, obj_paths, obj_scales, render_pc=False):
        """Render instance segmentation map

        Arguments:
            full_depth {np.ndarray} -- HxW depth map
            obj_paths {list} -- list of object paths in scene
            obj_scales {list} -- list of object scales in scene

        Keyword Arguments:
            render_pc {bool} -- whether to return object-wise point clouds (default: {False})

        Returns:
            [np.ndarray, list, dict] -- integer segmap with 0=background, list of
                                        corresponding object names, dict of corresponding point clouds
        """

        scene_object_nodes = []
        for n in self._scene.get_nodes():
            if n.name not in ['camera', 'parent']:
                n.mesh.is_visible = False
                if n.name != 'table':
                    scene_object_nodes.append(n)

        obj_names = [path + '_' + str(scale) for path, scale in zip(obj_paths,obj_scales)]


        pcs = {}
        output = np.zeros(full_depth.shape, np.uint8)
        for n in scene_object_nodes:
            n.mesh.is_visible = True

            depth = self.renderer.render(self._scene)[1]
            mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0
            )
            if not np.any(mask):
                continue
            if np.any(output[mask] != 0):
                raise ValueError('wrong label')

            indices = [i+1 for i, x in enumerate(obj_names) if x == n.name]
            for i in indices:
                if not np.any(output==i):
                    print('')
                    output[mask] = i
                    break

            n.mesh.is_visible = False

            if render_pc:
                pcs[i] = self._to_pointcloud(depth*mask)

        for n in self._scene.get_nodes():
            if n.name not in ['camera', 'parent']:
                n.mesh.is_visible = True

        return output, ['BACKGROUND'] + obj_names, pcs

class PointCloudReader:
    """
    Class to load scenes, render point clouds and augment them during training

    Arguments:
        root_folder {str} -- acronym root folder
        batch_size {int} -- number of rendered point clouds per-batch default is 1

    Keyword Arguments:
        raw_num_points {int} -- Number of random/farthest point samples per scene (default: {20000})
        estimate_normals {bool} -- compute normals from rendered point cloud (default: {False})
        caching {bool} -- cache scenes in memory (default: {True})
        use_uniform_quaternions {bool} -- use uniform quaternions for camera sampling (default: {False})
        scene_obj_scales {list} -- object scales in scene (default: {None})
        scene_obj_paths {list} -- object paths in scene (default: {None})
        scene_obj_transforms {np.ndarray} -- object transforms in scene (default: {None})
        num_train_samples {int} -- training scenes (default: {None})
        num_test_samples {int} -- test scenes (default: {None})
        use_farthest_point {bool} -- use farthest point sampling to reduce point cloud dimension (default: {False})
        intrinsics {str} -- intrinsics to for rendering depth maps (default: {None})
        distance_range {tuple} -- distance range from camera to center of table (default: {(0.9,1.3)})
        elevation {tuple} -- elevation range (90 deg is top-down) (default: {(30,150)})
        pc_augm_config {dict} -- point cloud augmentation config (default: {None})
        depth_augm_config {dict} -- depth map augmentation config (default: {None})
    """
    def __init__(
            self,
            root_folder,
            batch_size=1,
            raw_num_points = 20000,
            estimate_normals = False,
            caching=True,
            use_uniform_quaternions=False,
            scene_obj_scales=None,
            scene_obj_paths=None,
            scene_obj_transforms=None,
            num_train_samples=None,
            num_test_samples=None,
            use_farthest_point = False,
            intrinsics = None,
            distance_range = (1.0,2.0),
            elevation = (30,150),
            pc_augm_config = None,
            depth_augm_config = None
    ):
        self._root_folder = root_folder
        self._batch_size = batch_size
        self._raw_num_points = raw_num_points
        self._caching = caching
        self._num_train_samples = num_train_samples
        self._num_test_samples = num_test_samples
        self._estimate_normals = estimate_normals
        self._use_farthest_point = use_farthest_point
        self._scene_obj_scales = scene_obj_scales
        self._scene_obj_paths = scene_obj_paths
        self._scene_obj_transforms = scene_obj_transforms
        self._distance_range = distance_range
        self._pc_augm_config = pc_augm_config
        self._depth_augm_config = depth_augm_config

        self._current_pc = None
        self._cache = {}

        self._renderer = SceneRenderer(caching=True, intrinsics=intrinsics)

        if use_uniform_quaternions:
            quat_path = os.path.join(self._root_folder, 'uniform_quaternions/data2_4608.qua')
            quaternions = [l[:-1].split('\t') for l in open(quat_path, 'r').readlines()]

            quaternions = [[float(t[0]),
                            float(t[1]),
                            float(t[2]),
                            float(t[3])] for t in quaternions]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._cam1_orientations = []
            self._cam2_orientations = []
            self._elevation = np.array(elevation)/180.
            for az in np.linspace(0, np.pi * 2, 30):
                for el in np.linspace(self._elevation[0], self._elevation[1], 30):
                    self._cam1_orientations.append(tra.euler_matrix(0, -el, az))
                    self._cam2_orientations.append(tra.euler_matrix(0, -el+np.random.uniform(-np.pi/18,-np.pi/18), az+np.pi+np.random.uniform(-np.pi/18,-np.pi/18)))
            self._coordinate_transform = tra.euler_matrix(np.pi/2, 0, 0).dot(tra.euler_matrix(0, np.pi/2, 0))

    def get_cam_pose(self, cam_orientation):
        """
        Samples camera pose on shell around table center

        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix

        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """

        distance = self._distance_range[0] + np.random.rand()*(self._distance_range[1]-self._distance_range[0])

        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance
        extrinsics = cam_orientation.dot(extrinsics)

        cam_pose = extrinsics.dot(self._coordinate_transform)
        # table height
        cam_pose[2,3] += self._renderer._table_dims[2]

        # I don't know why the author add this!
        # cam_pose[:3,:2]= -cam_pose[:3,:2]

        return cam_pose

    def get_two_cam_pose(self, cam1_orientation, cam2_orientation):
        """
        Samples camera pose on shell around table center

        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix

        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """

        distance = self._distance_range[0] + np.random.rand()*(self._distance_range[1]-self._distance_range[0])

        extrinsics1 = np.eye(4)
        extrinsics1[0, 3] += distance
        extrinsics1 = cam1_orientation.dot(extrinsics1)

        extrinsics2 = np.eye(4)
        extrinsics2[0, 3] += distance
        extrinsics2 = cam2_orientation.dot(extrinsics2)

        cam_pose1 = extrinsics1.dot(self._coordinate_transform)
        cam_pose2 = extrinsics2.dot(self._coordinate_transform)
        # table height
        cam_pose1[2,3] += self._renderer._table_dims[2]
        cam_pose2[2,3] += self._renderer._table_dims[2]

        # I don't know why the author add this!
        # cam_pose[:3,:2]= -cam_pose[:3,:2]

        return cam_pose1, cam_pose2

    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud

        Returns:
            np.ndarray -- augmented point cloud
        """

        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self.apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'],
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            pc = provider.jitter_point_cloud(pc[np.newaxis, :, :],
                                             sigma=self._pc_augm_config['sigma'],
                                             clip=self._pc_augm_config['clip'])[0]


        return pc[:,:3]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config

        Arguments:
            depth {np.ndarray} -- depth map

        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth,(kernel,kernel),0)
            depth[depth_copy==0] = depth_copy[depth_copy==0]

        return depth

    def apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- noof cluster to remove
            occlusion_dropout_rate {float} -- prob of removal

        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc

        labels = farthest_points(pc, occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def get_scene(self, scene_idx=None, return_segmap=False, save=True):
        """
        Render a batch of scene point clouds

        Keyword Arguments:
            scene_idx {int} -- index of the scene (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})

        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        dims = 6 if self._estimate_normals else 3
        batch_pc = np.empty((self._batch_size, self._raw_num_points*2, dims), dtype=np.float32)
        cam1_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)
        cam2_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)
        batch_depth1 = np.empty((self._batch_size, self._renderer._height, self._renderer._width), dtype=np.float32)
        batch_depth2 = np.empty((self._batch_size, self._renderer._height, self._renderer._width), dtype=np.float32)

        obj_paths = [os.path.join(self._root_folder, p) for p in self._scene_obj_paths]
        mesh_scales = self._scene_obj_scales
        obj_trafos = self._scene_obj_transforms

        self.change_scene(obj_paths, mesh_scales, obj_trafos, visualize=False)

        batch_segmap, batch_obj_pcs = [], []
        cam_pose_path = os.path.join(self._root_folder, 'cam_pose_all', '%06d'%(scene_idx)) # camera pose files
        if not os.path.exists(cam_pose_path):
            os.makedirs(cam_pose_path)

        for i in range(self._batch_size):
            # 0.005s camera1_pose opengl
            pc_cam, pc1_normals, pc2_normals, camera1_pose, camera2_pose, depth1, depth2 = self.render_random_scene(estimate_normals = self._estimate_normals)

            batch_pc[i,:,0:3] = pc_cam[:,:3]
            batch_depth1[i,:,:] = depth1
            batch_depth2[i,:,:] = depth2

            # deprecated estimate_normals not supported
            if self._estimate_normals:
                batch_pc[i,:,3:6] = pc1_normals[:,:3]

            cam1_poses[i,:,:] = camera1_pose
            cam2_poses[i,:,:] = camera2_pose

            if save:
                K = np.array([[self._renderer._fx,0,self._renderer._cx],[0,self._renderer._fy,self._renderer._cy],[0,0,1]])
                data = {'K':K, 'camera1_pose':camera1_pose, 'camera2_pose':camera2_pose, 'scene_idx':scene_idx}
                file_path = os.path.join(cam_pose_path, '{}_dual_opengl.npz'.format(str(i))) # info files
                np.savez(file_path, **data)


        return batch_pc, batch_depth1, batch_depth2, cam1_poses, cam2_poses, scene_idx

    def render_random_scene(self, estimate_normals=False, camera_pose=None):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations

        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {[type]} -- camera pose to render the scene from. (default: {None})

        Returns:
            [pc, pc_normals, camera1_pose, camera2_pose, depth1, depth2] -- [point cloud, point cloud normals, camera1 pose, camera2 pose, depth1, depth2]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._cam1_orientations))
            camera1_orientation = self._cam1_orientations[viewing_index]
            camera2_orientation = self._cam2_orientations[viewing_index]
            camera1_pose, camera2_pose = self.get_two_cam_pose(camera1_orientation, camera2_orientation)


        in_camera1_pose = copy.deepcopy(camera1_pose)
        in_camera2_pose = copy.deepcopy(camera2_pose)

        # 0.005 s
        _, depth1, _, camera1_pose = self._renderer.render(in_camera1_pose, render_pc=False)
        _, depth2, _, camera2_pose = self._renderer.render(in_camera2_pose, render_pc=False)

        depth1 = self._augment_depth(depth1)
        depth2 = self._augment_depth(depth2)


        pc1 = self._renderer._to_pointcloud(depth1)
        pc1 = regularize_pc_point_count(pc1, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        pc1 = self._augment_pc(pc1)
        pc1_normals = estimate_normals_cam_from_pc(pc1[:,:3], raw_num_points=self._raw_num_points) if estimate_normals else []

        pc2 = self._renderer._to_pointcloud(depth2)
        pc2 = regularize_pc_point_count(pc2, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        pc2 = self._augment_pc(pc2)
        pc2_normals = estimate_normals_cam_from_pc(pc2[:,:3], raw_num_points=self._raw_num_points) if estimate_normals else []


        # opengl to opencv
        in_camera1_pose[:3,1] = -in_camera1_pose[:3,1]
        in_camera1_pose[:3,2] = -in_camera1_pose[:3,2]
        in_camera2_pose[:3,1] = -in_camera2_pose[:3,1]
        in_camera2_pose[:3,2] = -in_camera2_pose[:3,2]

        rel_cam_pose = np.matmul(inverse_transform(in_camera1_pose), in_camera2_pose)



        pc = np.concatenate((pc1, (np.matmul(rel_cam_pose[:3, :3], pc2.T) + rel_cam_pose[:3, 3].reshape(3,-1)).T), axis=0) # pc_all



        return pc, pc1_normals, pc2_normals, camera1_pose, camera2_pose, depth1, depth2

    def change_object(self, cad_path, cad_scale):
        """
        Change object in pyrender scene

        Arguments:
            cad_path {str} -- path to CAD model
            cad_scale {float} -- scale of CAD model
        """

        self._renderer._load_object(cad_path, cad_scale)

    def change_scene(self, obj_paths, obj_scales, obj_transforms, visualize=False):
        """
        Change pyrender scene

        Arguments:
            obj_paths {list[str]} -- path to CAD models in scene
            obj_scales {list[float]} -- scales of CAD models
            obj_transforms {list[np.ndarray]} -- poses of CAD models

        Keyword Arguments:
            visualize {bool} -- whether to update the visualizer as well (default: {False})
        """
        self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        if visualize:
            self._visualizer.change_scene(obj_paths, obj_scales, obj_transforms)



    def __del__(self):
        print('********** terminating renderer **************')

class TableScene(Scene):
    """
    Holds current table-top scene, samples object poses and checks grasp collisions.

    Arguments:
        root_folder {str} -- path to acronym data
        gripper_path {str} -- relative path to gripper collision mesh

    Keyword Arguments:
        lower_table {float} -- lower table to permit slight grasp collisions between table and object/gripper (default: {0.02})
    """

    def __init__(self, root_folder, gripper_path, grasp_used_path, lower_table=0.02, splits=['train']):

        super().__init__()
        self.root_folder = root_folder
        self.splits= splits
        self.gripper_mesh = trimesh.load(os.path.join(BASE_DIR, gripper_path))

        self._table_dims = [4.0, 4.8, 0.6]
        self._table_support = [0.6, 0.6, 0.6]
        self._table_pose = np.eye(4)
        self.table_mesh = trimesh.creation.box(self._table_dims)
        self.table_support = trimesh.creation.box(self._table_support)
        self.grasp_used_path = grasp_used_path
        self.grasp_paths = load_grasp_path(root_folder)
        # self.data_splits = load_splits(root_folder)
        # self.category_list = list(self.data_splits.keys())
        # self.contact_infos = load_contacts(root_folder, self.data_splits, splits=self.splits)

        self._lower_table = lower_table

        self._scene_count = 0

    def get_random_object(self):

        """Return random scaled but not yet centered object mesh

        Returns:
            [trimesh.Trimesh, str] -- ShapeNet mesh from a random category, h5 file path
        """

        while True:
            random_category = random.choice(self.category_list)
            cat_obj_paths = [obj_p for split in self.splits for obj_p in self.data_splits[random_category][split]]
            if cat_obj_paths:
                random_grasp_path = random.choice(cat_obj_paths)
                if random_grasp_path in self.contact_infos:
                    break

        obj_mesh = load_mesh(os.path.join(self.root_folder, 'grasp', random_grasp_path), self.root_folder)

        # mesh_mean =  np.mean(obj_mesh.vertices, 0, keepdims=True)
        # obj_mesh.vertices -= mesh_mean

        return obj_mesh, random_grasp_path

    def get_random_object_withoutlabel(self):

        """Return random scaled but not yet centered object mesh

        Returns:
            [trimesh.Trimesh, str] -- ShapeNet mesh from a random category, h5 file path
        """

        random_grasp_path = random.choice(self.grasp_paths)
        print(random_grasp_path)


        obj_mesh = load_mesh(random_grasp_path, os.path.join(self.root_folder,'simplified'))

        # mesh_mean =  np.mean(obj_mesh.vertices, 0, keepdims=True)
        # obj_mesh.vertices -= mesh_mean

        return obj_mesh, random_grasp_path

    def _get_random_stable_pose(self, stable_poses, stable_poses_probs, thres=0.005):
        """Return a stable pose according to their likelihood.

        Args:
            stable_poses (list[np.ndarray]): List of stable poses as 4x4 matrices.
            stable_poses_probs (list[float]): List of probabilities.
            thres (float): Threshold of pose stability to include for sampling

        Returns:
            np.ndarray: homogeneous 4x4 matrix
        """


        # Random pose with unique (avoid symmetric poses) stability prob > thres
        _,unique_idcs = np.unique(stable_poses_probs.round(decimals=3), return_index=True)
        unique_idcs = unique_idcs[::-1]
        unique_stable_poses_probs = stable_poses_probs[unique_idcs]
        n = len(unique_stable_poses_probs[unique_stable_poses_probs>thres])
        index = unique_idcs[np.random.randint(n)]

        # index = np.random.choice(len(stable_poses), p=stable_poses_probs)
        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])

    def _get_stand_pose(self, stand_pose):


        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stand_pose)

    def find_object_placement(self, obj_mesh, max_iter, stand=True):
        """Try to find a non-colliding stable pose on top of any support surface.

        Args:
            obj_mesh (trimesh.Trimesh): Object mesh to be placed.
            max_iter (int): Maximum number of attempts to place to object randomly.
            stand: whether let the object "stand" on the table

        Raises:
            RuntimeError: In case the support object(s) do not provide any support surfaces.

        Returns:
            bool: Whether a placement pose was found.
            np.ndarray: Homogenous 4x4 matrix describing the object placement pose. Or None if none was found.
        """
        support_polys, support_T = self._get_support_polygons()
        if len(support_polys) == 0:
            raise RuntimeError("No support polygons found!")

        # get stable poses for object
        stable_obj = obj_mesh.copy()

        # stable_obj.vertices -= stable_obj.center_mass # TODO

        stable_poses, stable_poses_probs = stable_obj.compute_stable_poses(
            threshold=0, sigma=0, n_samples=20
        )
        # stable_poses, stable_poses_probs = obj_mesh.compute_stable_poses(threshold=0, sigma=0, n_samples=1)

        # Sample support index
        support_index = max(enumerate(support_polys), key=lambda x: x[1].area)[0]

        iter = 0
        colliding = True
        while iter < max_iter and colliding:

            # Sample position in plane
            pts = trimesh.path.polygons.sample(
                support_polys[support_index], count=1
            )

            # To avoid collisions with the support surface
            pts3d = np.append(pts, 0)

            # Transform plane coordinates into scene coordinates
            placement_T = np.dot(
                support_T[support_index],
                trimesh.transformations.translation_matrix(pts3d),
            )
            if stand:
                stand_pose = stable_poses[np.argmax(np.matmul(stable_poses[:,:3,2], np.array([0,0,1]))),:,:]
                pose = self._get_stand_pose(stand_pose)
            else:
                pose = self._get_random_stable_pose(stable_poses, stable_poses_probs)

            placement_T = np.dot(
                np.dot(placement_T, pose), tra.translation_matrix(-obj_mesh.center_mass)
            )

            # Check collisions
            colliding = self.is_colliding(obj_mesh, placement_T)

            iter += 1

        return not colliding, placement_T if not colliding else None

    def is_colliding(self, mesh, transform, eps=1e-6):
        """
        Whether given mesh collides with scene

        Arguments:
            mesh {trimesh.Trimesh} -- mesh
            transform {np.ndarray} -- mesh transform

        Keyword Arguments:
            eps {float} -- minimum distance detected as collision (default: {1e-6})

        Returns:
            [bool] -- colliding or not
        """
        dist = self.collision_manager.min_distance_single(mesh, transform=transform)
        return dist < eps

    def load_suc_obj_contact_grasps(self, grasp_path):
        """
        Loads successful object grasp contacts

        Arguments:
            grasp_path {str} -- acronym grasp path

        Returns:
            [np.ndarray, np.ndarray] -- Mx4x4 grasp transforms, Mx3 grasp contacts
        """
        contact_info = self.contact_infos[grasp_path]

        suc_grasps = contact_info['successful'].reshape(-1)
        gt_grasps = contact_info['grasp_transform'].reshape(-1,4,4)
        gt_contacts = contact_info['contact_points'].reshape(-1,3)

        suc_gt_contacts = gt_contacts[np.repeat(suc_grasps,2)>0]
        suc_gt_grasps = gt_grasps[suc_grasps>0]

        return suc_gt_grasps, suc_gt_contacts

    def load_suc_obj_dual_grasps(self, grasp_path):
        """
        Loads successful object grasp contacts

        Arguments:
            grasp_path {str} -- acronym grasp path

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- successful grasps (all of them), For measures, Tor measures, Dex measures.
        """
        grasp_info = h5py.File(grasp_path, "r")

        gt_grasps = np.array(grasp_info["grasps/transforms"])
        Force_closure = np.array(grasp_info["/grasps/qualities/Force_closure"])
        Torque_optimization = np.array(grasp_info["grasps/qualities/Torque_optimization"])
        Dexterity = np.array(grasp_info["grasps/qualities/Dexterity"])

        suc_gt_grasps = gt_grasps
        For_gt_grasps = Force_closure
        Tor_gt_grasps = Torque_optimization
        Dex_gt_grasps = Dexterity

        return suc_gt_grasps, For_gt_grasps, Tor_gt_grasps, Dex_gt_grasps

    def set_mesh_transform(self, name, transform):
        """
        Set mesh transform for collision manager

        Arguments:
            name {str} -- mesh name
            transform {np.ndarray} -- 4x4 homog mesh pose
        """
        self.collision_manager.set_transform(name, transform)
        self._poses[name] = transform

    def save_scene_grasps(self, output_dir, scene_filtered_grasps, scene_fors, scene_tors, scene_dexs, obj_paths, obj_transforms, obj_scales, obj_grasp_idcs):
        """
        Save scene_contact infos in output_dir

        Arguments:
            output_dir {str} -- absolute output directory
            scene_filtered_grasps {np.ndarray} -- Nx2x4x4 filtered scene grasps
            scene_fors {np.ndarray} -- Nx1 filtered for measures
            scene_tors {np.ndarray} -- Nx1 filtered tor measures
            scene_dexs {np.ndarray} -- Nx1 filtered dex measures
            obj_paths {list} -- list of object paths in scene
            obj_transforms {list} -- list of object transforms in scene
            obj_scales {list} -- list of object scales in scene
            obj_grasp_idcs {list} -- list of starting grasp idcs for each object
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        contact_info = {}
        contact_info['obj_paths'] = obj_paths
        contact_info['obj_transforms'] = obj_transforms
        contact_info['obj_scales'] = obj_scales
        contact_info['grasp_transforms'] = scene_filtered_grasps
        contact_info['grasp_qualities_fors'] = scene_fors
        contact_info['grasp_qualities_tors'] = scene_tors
        contact_info['grasp_qualities_dexs'] = scene_dexs
        contact_info['obj_grasp_idcs'] = np.array(obj_grasp_idcs)
        output_path = os.path.join(output_dir, '{:06d}.npz'.format(self._scene_count))
        while os.path.exists(output_path):
            self._scene_count += 1
            output_path = os.path.join(output_dir, '{:06d}.npz'.format(self._scene_count))
        np.savez(output_path, **contact_info)
        self._scene_count += 1

    def _transform_grasps(self, grasps, contacts, obj_transform):
        """
        Transform grasps and contacts into given object transform

        Arguments:
            grasps {np.ndarray} -- Nx4x4 grasps
            contacts {np.ndarray} -- 2Nx3 contacts
            obj_transform {np.ndarray} -- 4x4 mesh pose

        Returns:
            [np.ndarray, np.ndarray] -- transformed grasps and contacts
        """
        transformed_grasps = np.matmul(obj_transform, grasps)
        contacts_homog = np.concatenate((contacts, np.ones((contacts.shape[0], 1))),axis=1)
        transformed_contacts = np.dot(contacts_homog, obj_transform.T)[:,:3]
        return transformed_grasps, transformed_contacts

    def _transform_dual_grasps(self, grasps, obj_transform):
        """
        Transform grasps and contacts into given object transform

        Arguments:
            grasps {np.ndarray} -- Nx2x4x4 grasps
            obj_transform {np.ndarray} -- 4x4 mesh pose

        Returns:
            [np.ndarray, np.ndarray] -- transformed grasps
        """
        print(obj_transform.shape, grasps.shape)
        transformed_grasps = np.matmul(obj_transform, grasps)
        # transformed_grasps = np.concatenate((transformed_grasps1,transformed_grasps2),axis=1)

        return transformed_grasps

    def _filter_colliding_dual_grasps(self, transformed_grasps, For, tor, dex):
        """
        Filter out colliding grasps

        Arguments:
            transformed_grasps {np.ndarray} -- Nx2x4x4 grasps
            For {np.ndarray} -- Nx1 measures
            Tor {np.ndarray} -- Nx1 measures
            Dex {np.ndarray} -- Nx1 measures

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- Mx2x4x4 filtered grasps, Mx1 For,  Mx1 Tor,  Mx1 Dex
        """
        filtered_grasps = []
        filtered_fors = []
        filtered_tors = []
        filtered_dexs = []
        for i,g in enumerate(transformed_grasps):
            if self.is_colliding(self.gripper_mesh, g[0]) or self.is_colliding(self.gripper_mesh, g[1]):
                # print("collide: ", g[0][:3,:],g[1][:3,:])
                continue
            filtered_grasps.append(g)
            filtered_fors.append(For[i])
            filtered_tors.append(tor[i])
            filtered_dexs.append(dex[i])
            # print("no collide: ", g[0][:3,:],g[1][:3,:])
        return np.array(filtered_grasps).reshape(-1,2,4,4), np.array(filtered_fors).reshape(-1,1), np.array(filtered_tors).reshape(-1,1), np.array(filtered_dexs).reshape(-1,1)

    def reset(self):
        """
        Reset, i.e. remove scene objects
        """
        for name in self._objects:
            self.collision_manager.remove_object(name)
        self._objects = {}
        self._poses = {}
        self._support_objects = []

    def load_existing_scene(self, path):
        """
        Load an existing scene_contacts scene for visualization

        Arguments:
            path {str} -- abs path to scene_contacts npz file

        Returns:
            [np.ndarray, list, list] -- scene_grasps, list of obj paths, list of object transforms
        """
        self.add_object('table', self.table_mesh, self._table_pose)
        self._support_objects.append(self.table_support)

        inp = np.load(os.path.join(self.root_folder, path))
        scene_filtered_grasps = inp['grasp_transforms']
        obj_transforms = inp['obj_transforms']
        obj_paths = inp['obj_paths']
        obj_scales = inp['obj_scales']

        for obj_path,obj_transform,obj_scale in zip(obj_paths,obj_transforms,obj_scales):
            obj_mesh = trimesh.load(os.path.join(self.root_folder, obj_path))
            obj_mesh.apply_transform(RigidTransform(np.eye(3), -obj_mesh.centroid).matrix)
            obj_mesh = obj_mesh.apply_scale(obj_scale)
            # mesh_mean =  np.mean(obj_mesh.vertices, 0, keepdims=True)
            # obj_mesh.vertices -= mesh_mean
            self.add_object(obj_path, obj_mesh, obj_transform)
        return scene_filtered_grasps, obj_paths, obj_transforms, obj_scales


    def handler(self, signum, frame):
        raise Exception("Could not place object ")

    def arrange(self, num_obj, max_iter=100, time_out = 8):
        """
        Arrange random table top scene with contact grasp annotations

        Arguments:
            num_obj {int} -- number of objects

        Keyword Arguments:
            max_iter {int} -- maximum iterations to try placing an object (default: {100})
            time_out {int} -- maximum time to try placing an object (default: {8})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list, list] --
            scene_filtered_grasps, scene_filtered_fors, scene_filtered_tors, scene_filtered_dexs, obj_paths, obj_transforms, obj_scales, obj_grasp_idcs

        """

        self._table_pose[2,3] -= self._lower_table
        self.add_object('table', self.table_mesh, self._table_pose)

        self._support_objects.append(self.table_support)

        obj_paths = []
        obj_transforms = []
        obj_scales = []
        grasp_paths = []

        for i in range(num_obj):
            obj_mesh, random_grasp_path = self.get_random_object_withoutlabel()
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(8)
            try:
                success, placement_T = self.find_object_placement(obj_mesh, max_iter)
            except Exception as exc:
                print(exc, random_grasp_path, " after {} seconds!".format(time_out))
                continue
            signal.alarm(0)
            if success:
                self.add_object(random_grasp_path, obj_mesh, placement_T)
                obj_scales.append(float(random_grasp_path.split('_')[-1].split('.h5')[0])) # scales
                obj_paths.append(os.path.join('simplified', '/'.join(random_grasp_path.split('/')[-1].split('_')[:1]) + '.obj')) # mesh files
                obj_transforms.append(placement_T)
                grasp_paths.append(random_grasp_path)
            else:
                print("Couldn't place object", random_grasp_path, " after {} iterations!".format(max_iter))
        print('Placed {} objects'.format(len(obj_paths)))

        # self.set_mesh_transform('table', self._table_pose)

        scene_filtered_grasps = []
        scene_filtered_fors = []
        scene_filtered_tors = []
        scene_filtered_dexs = []
        obj_grasp_idcs = []
        grasp_count = 0

        for obj_transform, grasp_path in zip(obj_transforms, grasp_paths):
            grasps, For, Tor, Dex = self.load_suc_obj_dual_grasps(grasp_path)
            if not grasps.shape[0]:
                continue
            transformed_grasps = self._transform_dual_grasps(grasps, obj_transform)
            filtered_grasps, filtered_fors, filtered_tors, filtered_dexs = self._filter_colliding_dual_grasps(transformed_grasps, For, Tor, Dex)

            scene_filtered_grasps.append(filtered_grasps)
            scene_filtered_fors.append(filtered_fors)
            scene_filtered_tors.append(filtered_tors)
            scene_filtered_dexs.append(filtered_dexs)
            obj_grasp_idcs.append(grasp_count)

        if scene_filtered_grasps:
            print(len(scene_filtered_grasps[0]))
            print(len(scene_filtered_tors[0]))
            scene_filtered_grasps = np.concatenate(scene_filtered_grasps,0)
            scene_filtered_fors = np.concatenate(scene_filtered_fors,0)
            scene_filtered_tors = np.concatenate(scene_filtered_tors,0)
            scene_filtered_dexs = np.concatenate(scene_filtered_dexs,0)

        self._table_pose[2,3] += self._lower_table
        self.set_mesh_transform('table', self._table_pose)

        return scene_filtered_grasps, scene_filtered_fors, scene_filtered_tors, scene_filtered_dexs, obj_paths, obj_transforms, obj_scales, obj_grasp_idcs

    def visualize_dual(self, pcreader, cam1_pose, cam2_pose, scene_grasps):
        """
        Visualizes table top scene with grasps

        Arguments:
            scene_grasps {np.ndarray} -- Nx4x4 grasp transforms
            scene_contacts {np.ndarray} -- Nx2x3 grasp contacts
        """
        successful_grasps = []
        marker = []
        database = []
        wave = len(scene_grasps)//3
        def countX(lst, x):
            count = 0
            for ele in lst:
                if (ele == x).all() :
                    count = count + 1
            return count
        print('Visualizing scene and grasps.. takes time')
        for i, (t1, t2) in enumerate(scene_grasps[0:50]):

            current_t1 = countX(database, t1)
            current_t2 = countX(database, t2)
            color = i/wave*255 if wave != 0 else 255
            code1 = color if color<=255 else 0
            code2 = color%255 if color>255 and color<=510 else 0
            code3 = color%510 if color>510 and color<=765 else 0
            successful_grasps.append((create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t1), create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t2)))

            trans1 = t1.dot(np.array([0,-0.067500/2-0.02*current_t1,0,1]).reshape(-1,1))[0:3]
            trans2 = t2.dot(np.array([0,-0.067500/2-0.02*current_t2,0,1]).reshape(-1,1))[0:3]

            tmp1 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans1).matrix)
            tmp1.visual.face_colors = [code1, code2, code3]
            tmp2 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans2).matrix)
            tmp2.visual.face_colors = [code1, code2, code3]
            marker.append(tmp1.copy())
            marker.append(tmp1.copy())
            database.append(t1)
            database.append(t2)

        trimesh_scene = self.colorize().as_trimesh_scene()
        trimesh_camera = pcreader._renderer.get_trimesh_camera()

        # cam_pose in opencv
        trimesh_scene.add_geometry(
            trimesh.creation.camera_marker(trimesh_camera),
            node_name="camera1",
            transform=cam1_pose,
        )
        trimesh_scene.add_geometry(
            trimesh.creation.camera_marker(trimesh_camera),
            node_name="camera2",
            transform=cam2_pose,
        )

        # show scene together with successful and collision-free grasps of all objects
        trimesh.scene.scene.append_scenes(
            [trimesh_scene, trimesh.Scene(successful_grasps[0:50] + marker[0:50])]
        ).show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Grasp data reader")
    parser.add_argument('root_folder', help='Root dir with grasps, meshes and splits', type=str)
    parser.add_argument('--num_grasp_scenes', type=int, default=10000)
    parser.add_argument('--splits','--list', nargs='+')
    parser.add_argument('--max_iterations', type=int, default=100)
    parser.add_argument('--gripper_path', type=str, default='scripts/grippers/robotiq_85/gripper.obj',help='gripper file')
    parser.add_argument('--config_path', type=str, default='./scene.yaml')
    parser.add_argument('--min_num_objects', type=int, default=1)
    parser.add_argument('--max_num_objects', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1,help=' number of views')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--load_existing', type=str, default=None,help=' all / number of the scene ')
    parser.add_argument('--grasps_used', type=str, default='grasp_used')
    parser.add_argument('--output_dir', type=str, default='table_scene_stand_all')
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    root_folder = args.root_folder
    splits = args.splits if args.splits else ['train','test']
    max_iterations = args.max_iterations
    gripper_path = args.gripper_path
    number_of_scenes = args.num_grasp_scenes
    min_num_objects = args.min_num_objects
    max_num_objects = args.max_num_objects
    start_index = args.start_index
    load_existing = args.load_existing
    output_dir = args.output_dir
    visualize = args.vis
    config_path = args.config_path
    grasp_used_path = os.path.join(root_folder, args.grasps_used)
    if not os.path.isdir(grasp_used_path):
        os.mkdir(grasp_used_path)

    table_scene = TableScene(root_folder, gripper_path, grasp_used_path, splits=splits)
    table_scene._scene_count = start_index

    print('Root folder', args.root_folder)
    output_dir = os.path.join(root_folder, output_dir)
    if load_existing == "all":
        path = glob.glob(os.path.join(root_folder, args.output_dir, "*.npz"))
        path.sort()
    elif load_existing:
        path = glob.glob(os.path.join(root_folder, args.output_dir, "{}.npz".format(load_existing.zfill(6))))
    else:
        path = []
    i = 0

    while table_scene._scene_count < number_of_scenes or i < len(path):
        if load_existing is not None:
            if i >= len(path):
                break
        table_scene.reset()

        if load_existing is None:
            # generating new scenes
            print('generating %s/%s' % (table_scene._scene_count, number_of_scenes))
            num_objects = np.random.randint(min_num_objects,max_num_objects+1)
            scene_grasps, scene_fors, scene_tors, scene_dexs, obj_paths, obj_transforms, obj_scales, obj_grasp_idcs = table_scene.arrange(num_objects, max_iterations)
            if not visualize:
                if len(scene_grasps):
                    table_scene.save_scene_grasps(output_dir, scene_grasps, scene_fors, scene_tors, scene_dexs, obj_paths, obj_transforms, obj_scales, obj_grasp_idcs)
                else:
                    continue
        else:
            # load old scenes
            scene_grasps, obj_paths, obj_transforms, obj_scales = table_scene.load_existing_scene(path[i])


        with open(config_path,'r') as f:
            global_config = yaml.safe_load(f)

        pcreader = PointCloudReader(
            root_folder=global_config['DATA']['data_path'],
            batch_size=args.batch_size,
            estimate_normals=global_config['DATA']['input_normals'],
            raw_num_points=global_config['DATA']['raw_num_points'],
            use_uniform_quaternions = global_config['DATA']['use_uniform_quaternions'],
            scene_obj_scales = obj_scales,
            scene_obj_paths = obj_paths,
            scene_obj_transforms = obj_transforms,
            num_train_samples = None,
            num_test_samples = None,
            use_farthest_point = global_config['DATA']['use_farthest_point'],
            intrinsics=global_config['DATA']['intrinsics'],
            elevation=global_config['DATA']['view_sphere']['elevation'],
            distance_range=global_config['DATA']['view_sphere']['distance_range'],
            pc_augm_config=global_config['DATA']['pc_augm'],
            depth_augm_config=global_config['DATA']['depth_augm']
        )

        batch_data, batch_depth1, batch_depth2, cam1_poses, cam2_poses, scene_idx = pcreader.get_scene(scene_idx=table_scene._scene_count-1) if load_existing is None else pcreader.get_scene(scene_idx=int(path[i].split('/')[-1].split('.npz')[0]))

        # OpenCV OpenGL conversion
        cam1_poses, cam1_poses_inv, batch_data = center_pc_convert_cam(cam1_poses, batch_data)
        cam2_poses, cam2_poses_inv, batch_data = center_pc_convert_cam(cam2_poses, batch_data)
        npy_fname = os.path.join(root_folder, 'pc_two_view_all', path[i].split('/')[-1].split('.npz')[0]) if load_existing else os.path.join(root_folder, 'pc_two_view_all', '%06d'%(table_scene._scene_count-1))
        if not os.path.exists(npy_fname):
            os.makedirs(npy_fname)
        for x in range(len(batch_data)):
            pc_path = os.path.join(npy_fname, '{}.npy'.format(str(x)))
            np.save(pc_path, batch_data[x, :, :3])
        i += 1

        if visualize:
            for cam1_pose, cam2_pose in zip(cam1_poses, cam2_poses):
                # show point cloud
                mlab.figure(bgcolor=(1,1,1))
                mlab.points3d(0, 0, 0, scale_factor=0.5, color=(1,0,0))
                mlab.points3d(batch_data[0][:,0], batch_data[0][:,1], batch_data[0][:,2], scale_factor=0.05, color=(1,0,0))
                mlab.show()

                # plot everything except point cloud
                f, axarr = plt.subplots(1, 1)
                im = axarr.imshow(batch_depth1[0,:,:])
                f.colorbar(im, ax=axarr)
                plt.show()

                table_scene.visualize_dual(pcreader, cam1_pose, cam2_pose, scene_grasps)
            table_scene._scene_count +=1









