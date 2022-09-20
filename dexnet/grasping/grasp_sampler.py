"""
Classes for sampling grasps.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import copy
import IPython
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random
import sys
import time
import math
import trimesh.proximity as tp
import trimesh.sample as ts
import trimesh.collision as tc
import multiprocessing as mp
from threading import Thread


APPROACH_OFFSET = 1e-3 # offset for suction contact approach

USE_OPENRAVE = True
try:
    import openravepy as rave
except:
    logging.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False

# from visualization import Visualizer3D as vis3d
import scipy.stats as stats
import trimesh

from autolab_core import RigidTransform
from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, PointGraspMetrics3D, GraspableObject3D, OpenRaveCollisionChecker, GraspCollisionChecker
from DA2_tools import create_robotiq_marker
from itertools import combinations

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

class GraspSampler:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.
    Attributes
    ----------
    gripper : :obj:`RobotGripper`
        the gripper to compute grasps for
    config : :obj:`YamlConfig`
        configuration for the grasp sampler
    """
    __metaclass__ = ABCMeta

    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """ Configures the grasp generator."""
        self.friction_coef = config['sampling_friction_coef']
        self.min_friction_coef = config['sampling_friction_coef']
        self.friction_coef_inc = config['sampling_friction_coef_inc']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.target_num_grasps = config['target_num_grasps']
        if self.target_num_grasps is None:
            self.target_num_grasps = config['min_num_grasps']

        self.min_contact_dist = config['min_contact_dist']
        self.num_grasp_rots = 0
        if 'coll_check_num_grasp_rots' in config.keys():
            self.num_grasp_rots = config['coll_check_num_grasp_rots']
        self.max_num_surface_points = 100
        if 'max_num_surface_points' in config.keys():
            self.max_num_surface_points = config['max_num_surface_points']
        if 'grasp_dist_thresh' in config.keys():
            self.grasp_dist_thresh_ = config['grasp_dist_thresh']
        else:
            self.grasp_dist_thresh_ = 0
        if 'grasp_dist_alpha' in config.keys():
            self.grasp_dist_alpha = config['grasp_dist_alpha']
        else:
            self.grasp_dist_alpha = 0.005
        self.sigma_center = 0
        if 'sigma_center' in config.keys():
            self.sigma_center = config['sigma_center']
        self.sigma_axis = 0
        if 'sigma_axis' in config.keys():
            self.sigma_axis = config['sigma_axis']
        self.check_collisions = config['check_collisions']

        if 'approach_dist' in config.keys():
            self.approach_dist = config['approach_dist']
        if 'delta_approach' in config.keys():
            self.delta_approach = config['delta_approach']

        if 'sampling_reject_angle_threshold' in config.keys():
            self.sampling_reject_angle_threshold = config['sampling_reject_angle_threshold']
        if 'sampling_reject_edge_distance' in config.keys():
            self.sampling_reject_edge_distance = config['sampling_reject_edge_distance']
        if 'sampling_reject_use_extra_checking' in config.keys():
            self.sampling_reject_use_extra_checking = config['sampling_reject_use_extra_checking']
        else:
            self.sampling_reject_use_extra_checking = False

    @abstractmethod
    def sample_grasps(self, graspable):
        """
        Create a list of candidate grasps for a given object.
        Must be implemented for all grasp sampler classes.
        Parameters
        ---------
        graspable : :obj:`GraspableObject3D`
            object to sample grasps on
        """
        pass

    @abstractmethod
    def sample_hierachical_grasps(self, graspable, num_grasps, gripper=None,
                                  vis=False):
        pass

    def generate_grasps_stable_poses(self, graspable, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                                     sample_approach_angles=False, check_collisions=False, vis=False, config=None, **kwargs):
        """Samples a set of grasps for an object, aligning the approach angles to the object stable poses.
        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        stable_poses : :obj:`list` of :obj:`meshpy.StablePose`
            list of stable poses for the object with ids read from the database
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        check_collisions : bool
            whether or not to check collisions
        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # sample dense grasps
        scale, unaligned_grasps = self.generate_grasps(graspable, target_num_grasps=target_num_grasps, grasp_gen_mult=grasp_gen_mult,
                                                       max_iter=max_iter, check_collisions=check_collisions, vis=vis, config=config)

        # align for each stable pose
        # grasps = {}
        # for stable_pose in stable_poses:
        #     grasps[stable_pose.id] = []
        #     for grasp in unaligned_grasps:
        #         aligned_grasp = grasp.perpendicular_table(grasp)
        #         grasps[stable_pose.id].append(copy.deepcopy(aligned_grasp))
        grasps = []
        for grasp in unaligned_grasps:
            aligned_grasp0 = grasp[0].perpendicular_table(grasp[0])
            aligned_grasp1 = grasp[1].perpendicular_table(grasp[1])
            grasps.append(copy.deepcopy((aligned_grasp0, aligned_grasp1)))


        return scale, grasps

    def visualize(self, T, obj_mesh):
        database = []
        successful_grasps = []
        marker = []

        def countX(lst, x):
            count = 0
            for ele in lst:
                if (ele == x).all():
                    count = count + 1
            return count

        wave = len(T) // 3
        for i, (T1, T2) in enumerate(T[:4000]):
            # print("t1: ", t1)
            # print("t2: ", t2)
            t1 = (T1.gripper_pose(self.gripper) * self.gripper.T_mesh_gripper.inverse()).matrix
            t2 = (T2.gripper_pose(self.gripper) * self.gripper.T_mesh_gripper.inverse()).matrix

            current_t1 = countX(database, t1)
            current_t2 = countX(database, t2)
            color = i / wave * 255
            code1 = color if color <= 255 else 0
            code2 = color % 255 if color > 255 and color <= 510 else 0
            code3 = color % 510 if color > 510 and color <= 765 else 0
            successful_grasps.append((create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t1),
                                      create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t2)))

            trans1 = t1.dot(np.array([0, -0.067500 / 2 - 0.02 * current_t1, 0, 1]).reshape(-1, 1))[0:3]
            trans2 = t2.dot(np.array([0, -0.067500 / 2 - 0.02 * current_t2, 0, 1]).reshape(-1, 1))[0:3]

            tmp1 = trimesh.creation.icosphere(radius=0.01).apply_transform(RigidTransform(np.eye(3), trans1).matrix)
            tmp1.visual.face_colors = [code1, code2, code3]
            tmp2 = trimesh.creation.icosphere(radius=0.01).apply_transform(RigidTransform(np.eye(3), trans2).matrix)
            tmp2.visual.face_colors = [code1, code2, code3]
            marker.append(copy.deepcopy(tmp1))
            marker.append(copy.deepcopy(tmp2))
            database.append(t1)
            database.append(t2)

        trimesh.Scene([obj_mesh] + successful_grasps + marker).show()

    def generate_grasps(self, graspable, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, check_collisions=False, vis=False, config=None, **kwargs):
        """Samples a set of grasps for an object.
        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        check_collisions : bool
            whether or not to check collisions
        Return
        ------
        scale, f_final, d_final, t_final, final_grasps
        """
        params={'friction_coef': config['sampling_friction_coef']}

        # setup collision checking
        if USE_OPENRAVE and check_collisions:
            rave.raveSetDebugLevel(rave.DebugLevel.Error)
            collision_checker = OpenRaveCollisionChecker(self.gripper, view=False)
            collision_checker.set_object(graspable)

        # get num grasps
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps
        self.friction_coef = self.min_friction_coef

        grasps = []
        final_grasps = np.array([]).reshape(-1,2)
        d_list = []
        t_list = []
        f_list = []


        k = 1
        tmp = graspable.mesh.trimesh
        x_err,y_err,z_err = tmp.bounds[1,:]-tmp.bounds[0,:]
        scale = 1
        if max(x_err,y_err,z_err) > 1.0:
            max_scale = 1 / max(x_err,y_err,z_err)
            min_scale = 0.6 / max(x_err,y_err,z_err)
            scale = random.uniform(min_scale, max_scale)
        print('Scale: ', scale)
        graspable.mesh.trimesh.apply_transform(RigidTransform(np.eye(3), -graspable.mesh.trimesh.centroid).matrix)
        graspable.mesh.trimesh.apply_scale(scale)

        while num_grasps_remaining > 0 and k <= max_iter:
            print("{}/{} starts!!!!!!!".format(k,max_iter))
            # SAMPLING: generate more than we need
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining
            new_grasps = self.sample_hierachical_grasps(graspable, num_grasps_generate, gripper = self.gripper,
                                                        vis=False, **kwargs)

            # COVERAGE REJECTION: prune grasps by distance
            pruned_grasps = []
            for grasp in new_grasps:

                min_dist = np.inf
                for cur_grasp in grasps:
                    if isinstance(cur_grasp, ParallelJawPtGrasp3D):
                        dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp, alpha=self.grasp_dist_alpha)
                    if dist < min_dist:
                        min_dist = dist
                for cur_grasp in pruned_grasps:
                    if isinstance(cur_grasp, ParallelJawPtGrasp3D):
                        dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp, alpha=self.grasp_dist_alpha)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= self.grasp_dist_thresh_:
                    pruned_grasps.append(grasp)
            coll_free_grasps = pruned_grasps

            # add to the current grasp set
            grasps += coll_free_grasps
            print('Obtain %d single grasps'%(len(coll_free_grasps)))
            dual_grasp = np.array([c for c in combinations(coll_free_grasps, 2)])
            np.random.shuffle(dual_grasp)

            grasp1_center = np.array([grasp1.center for i, (grasp1, grasp2) in enumerate(dual_grasp)])
            grasp2_center = np.array([grasp2.center for i, (grasp1, grasp2) in enumerate(dual_grasp)])
            grasp1_axis = np.array([grasp1.axis for i, (grasp1, grasp2) in enumerate(dual_grasp)])
            grasp2_axis = np.array([grasp2.axis for i, (grasp1, grasp2) in enumerate(dual_grasp)])

            if not len(dual_grasp):
                k += 1
                continue
            dist = ParallelJawPtGrasp3D.array_distance(grasp1_center, grasp2_center, grasp1_axis, grasp2_axis, alpha=self.grasp_dist_alpha)
            print("skip  ", np.where(dist < 0.05)[0], "len: ", np.where(dist < 0.05)[0].shape)
            dual_grasp = dual_grasp[np.where(dist > 0.05)[0]]

            if not len(dual_grasp):
                k += 1
                continue
            grasp_points = np.array([[grasp1.grasp_point1, grasp1.grasp_point2, grasp2.grasp_point1, grasp2.grasp_point2] for i, (grasp1, grasp2) in enumerate(dual_grasp)])
            grasp_normals = np.array([[grasp1.grasp_normal1, grasp1.grasp_normal2, grasp2.grasp_normal1, grasp2.grasp_normal2] for i, (grasp1, grasp2) in enumerate(dual_grasp)])
            a_array, f_array = PointGraspMetrics3D.Dual_force_closure_batch(grasp_points, -grasp_normals, params=params)
            vaild_idx = np.where(a_array * 1e3 != 0)[0]
            print("Only %d grasp pairs pass the force closure "%len(vaild_idx))
            f_array = 1 - f_array[vaild_idx]
            d_array = PointGraspMetrics3D.Dexterity(grasp_points[vaild_idx])
            t_array = PointGraspMetrics3D.Torque_optimization(grasp_points[vaild_idx])
            final_grasps = np.concatenate((final_grasps, dual_grasp[vaild_idx]), axis=0)
            f_list = np.concatenate((f_list, f_array), axis=0)
            d_list = np.concatenate((d_list, d_array), axis=0)
            t_list = np.concatenate((t_list, t_array), axis=0)
            print(" Over all %d grasps have passed the force closure " % len(final_grasps))

            print('{}/{} dual grasps found after iteration {}.'.format(len(final_grasps), target_num_grasps, k))
            num_grasps_remaining = target_num_grasps - len(final_grasps)
            k += 1

        if final_grasps.size:
            f_final = np.array(f_list)
            d_final = np.array(d_list) * 2
            t_final = np.array(t_list)
        else:
            f_final = np.array([])
            d_final = np.array([])
            t_final = np.array([])

        option = 'number'
        if len(final_grasps) > target_num_grasps:
            if option == 'number':
                tmp = copy.deepcopy(f_final)
                tmp.sort()
                print('Truncating {} grasps to {}.'.format( len(final_grasps), target_num_grasps))
                # idx = np.argpartition(d_final, -target_num_grasps)[-target_num_grasps:] # Top target_num grasps
                f_final_best = np.unique(tmp[-len(f_final)//3:])
                f_final_medium = np.unique(tmp[-len(f_final)*2//3:-len(f_final)//3])
                f_final_worst = np.unique(tmp[0:len(f_final)//3])
                idx_best = np.unique(np.where(f_final==f_final_best[:,None])[-1])
                idx_medium = np.unique(np.where(f_final==f_final_medium[:,None])[-1])  # Middle target_num grasps
                idx_worst = np.unique(np.where(f_final==f_final_worst[:,None])[-1])  # Middle target_num grasps
                final_grasps = np.concatenate((final_grasps[idx_best][-target_num_grasps//3:], final_grasps[idx_medium][-target_num_grasps//3:], final_grasps[idx_worst][-target_num_grasps//3:]),axis=0)
                f_final = np.concatenate((f_final[idx_best][-target_num_grasps//3:], f_final[idx_medium][-target_num_grasps//3:], f_final[idx_worst][-target_num_grasps//3:]),axis=0)
                d_final = np.concatenate((d_final[idx_best][-target_num_grasps//3:], d_final[idx_medium][-target_num_grasps//3:], d_final[idx_worst][-target_num_grasps//3:]),axis=0)
                t_final = np.concatenate((t_final[idx_best][-target_num_grasps//3:], t_final[idx_medium][-target_num_grasps//3:], t_final[idx_worst][-target_num_grasps//3:]),axis=0)
            else:
                low, up = 0.7, 0.9
                idx_best = np.where(f_final > up)[0]
                idx_medium = np.where((f_final < up) & (f_final>low))[0]
                idx_worst = np.where(f_final < low)[0]
                final_grasps = np.concatenate((final_grasps[idx_best][-target_num_grasps//3:], final_grasps[idx_medium][-target_num_grasps//3:], final_grasps[idx_worst][-target_num_grasps//3:]),axis=0)
                f_final = np.concatenate((f_final[idx_best][-target_num_grasps // 3:],
                                          f_final[idx_medium][-target_num_grasps // 3:],
                                          f_final[idx_worst][-target_num_grasps // 3:]), axis=0)
                d_final = np.concatenate((d_final[idx_best][-target_num_grasps // 3:],
                                          d_final[idx_medium][-target_num_grasps // 3:],
                                          d_final[idx_worst][-target_num_grasps // 3:]), axis=0)
                t_final = np.concatenate((t_final[idx_best][-target_num_grasps // 3:],
                                          t_final[idx_medium][-target_num_grasps // 3:],
                                          t_final[idx_worst][-target_num_grasps // 3:]), axis=0)
                if len(final_grasps) < target_num_grasps:
                    res = target_num_grasps - len(final_grasps)
                    final_grasps = np.concatenate((final_grasps, final_grasps[idx_best][-target_num_grasps//3-res:-target_num_grasps//3]), axis=0)
                    f_final = np.concatenate((f_final,
                                              f_final[idx_best][-target_num_grasps//3-res:-target_num_grasps//3]), axis=0)
                    d_final = np.concatenate((d_final,
                                              d_final[idx_best][-target_num_grasps//3-res:-target_num_grasps//3]), axis=0)
                    t_final = np.concatenate((t_final,
                                              t_final[idx_best][-target_num_grasps//3-res:-target_num_grasps//3]), axis=0)

        print('Found {} grasps.'.format(len(final_grasps)))

        return scale, f_final, d_final, t_final, final_grasps

class UniformGraspSampler(GraspSampler):
    """ Sample grasps by sampling pairs of points on the object surface uniformly at random.
    """
    def sample_grasps(self, graspable, num_grasps,
                      vis=False, max_num_samples=1000):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]
        i = 0
        grasps = []

        # get all grasps
        while len(grasps) < num_grasps and i < max_num_samples:
            # get candidate contacts
            indices = np.random.choice(num_surface, size=2, replace=False)
            c0 = surface_points[indices[0], :]
            c1 = surface_points[indices[1], :]

            if np.linalg.norm(c1 - c0) > self.gripper.min_width and np.linalg.norm(c1 - c0) < self.gripper.max_width:
                # compute centers and axes
                grasp_center = ParallelJawPtGrasp3D.center_from_endpoints(c0, c1)
                grasp_axis = ParallelJawPtGrasp3D.axis_from_endpoints(c0, c1)
                g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center,
                                                                                        grasp_axis,
                                                                                        self.gripper.max_width))
                # keep grasps if the fingers close
                success, contacts = g.find_contacts(graspable)
                if success:
                    grasps.append(g)
            i += 1

        return grasps

class GaussianGraspSampler(GraspSampler):
    """ Sample grasps by sampling a center from a gaussian with mean at the object center of mass
    and grasp axis by sampling the spherical angles uniformly at random.
    """
    def sample_grasps(self, graspable, num_grasps,
                      vis=False,
                      sigma_scale=2.5):
        """
        Returns a list of candidate grasps for graspable object by Gaussian with
        variance specified by principal dimensions.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        sigma_scale : float
            the number of sigmas on the tails of the Gaussian for each dimension

        Returns
        -------
        :obj:`list` of obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get object principal axes
        center_of_mass = graspable.mesh.center_of_mass
        principal_dims = graspable.mesh.principal_dims()
        sigma_dims = principal_dims / (2 * sigma_scale)

        # sample centers
        grasp_centers = stats.multivariate_normal.rvs(
            mean=center_of_mass, cov=sigma_dims**2, size=num_grasps)

        # samples angles uniformly from sphere
        u = stats.uniform.rvs(size=num_grasps)
        v = stats.uniform.rvs(size=num_grasps)
        thetas = 2 * np.pi * u
        phis = np.arccos(2 * v - 1.0)
        grasp_dirs = np.array([np.sin(phis) * np.cos(thetas), np.sin(phis) * np.sin(thetas), np.cos(phis)])
        grasp_dirs = grasp_dirs.T

        # convert to grasp objects
        grasps = []
        for i in range(num_grasps):
            grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i,:], grasp_dirs[i,:], self.gripper.max_width))
            contacts_found, contacts = grasp.find_contacts(graspable)

            # add grasp if it has valid contacts
            if contacts_found and np.linalg.norm(contacts[0].point - contacts[1].point) > self.min_contact_dist:
                grasps.append(grasp)

        # visualize
        if vis:
            for grasp in grasps:
                plt.clf()
                h = plt.gcf()
                plt.ion()
                grasp.find_contacts(graspable, vis=vis)
                plt.show(block=False)
                time.sleep(0.5)

            grasp_centers_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_centers.T)
            grasp_centers_grid = grasp_centers_grid.T
            com_grid = graspable.sdf.transform_pt_obj_to_grid(center_of_mass)

            plt.clf()
            ax = plt.gca(projection = '3d')
            graspable.sdf.scatter()
            ax.scatter(grasp_centers_grid[:,0], grasp_centers_grid[:,1], grasp_centers_grid[:,2], s=60, c=u'm')
            ax.scatter(com_grid[0], com_grid[1], com_grid[2], s=120, c=u'y')
            ax.set_xlim3d(0, graspable.sdf.dims_[0])
            ax.set_ylim3d(0, graspable.sdf.dims_[1])
            ax.set_zlim3d(0, graspable.sdf.dims_[2])
            plt.show()

        return grasps

class SdfAntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone, then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """
    def sample_from_cone(self, n, tx, ty, num_samples=1):
        """ Samples directions from within the friction cone using uniform sampling.

        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector
        num_samples : int
            number of directions to sample

        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
       """
        v_samples = []
        for i in range(num_samples):
            theta = 2 * np.pi * np.random.rand()
            r = self.friction_coef * np.random.rand()
            v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
            v = -v / np.linalg.norm(v)
            v_samples.append(v)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Checks whether or not a direction is in the friction cone.
        This is equivalent to whether a grasp will slip using a point contact model.

        Parameters
        ----------
        cone : 3xN :obj:`numpy.ndarray`
            supporting vectors of the friction cone
        n : 3x1 :obj:`numpy.ndarray`
            outward pointing surface normal vector at c1
        v : 3x1 :obj:`numpy.ndarray`
            direction vector

        Returns
        -------
        in_cone : bool
            True if alpha is within the cone
        alpha : float
            the angle between the normal and v
        """
        if (v.dot(cone) < 0).any(): # v should point in same direction as cone
            v = -v # don't worry about sign, we don't know it anyway...
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point """
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def sample_grasps(self, graspable, num_grasps,
                      vis=False):
        """Returns a list of candidate grasps for graspable object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            number of grasps to sample
        vis : bool
            whether or not to visualize progress, for debugging

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            the sampled grasps
        """
        # get surface points
        grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        shuffled_surface_points = surface_points[:min(self.max_num_surface_points, len(surface_points))]
        logging.info('Num surface: %d' %(len(surface_points)))
        params={'friction_coef':self.friction_coef}

        for k, x_surf in enumerate(shuffled_surface_points):
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(self.num_samples):
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)

                # compute friction cone faces
                c1 = Contact3D(graspable, x1, in_direction=None)
                _, tx1, ty1 = c1.tangents()
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(n1, tx1, ty1, num_samples=1)
                sample_time = time.clock()

                for v in v_samples:
                    if vis:
                        x1_grid = graspable.sdf.transform_pt_obj_to_grid(x1)
                        cone1_grid = graspable.sdf.transform_pt_obj_to_grid(cone1, direction=True)
                        plt.clf()
                        h = plt.gcf()
                        plt.ion()
                        ax = plt.gca(projection = '3d')
                        for i in range(cone1.shape[1]):
                            ax.scatter(x1_grid[0] - cone1_grid[0], x1_grid[1] - cone1_grid[1], x1_grid[2] - cone1_grid[2], s = 50, c = u'm')

                    # random axis flips since we don't have guarantees on surface normal directoins
                    if random.random() > 0.5:
                        v = -v

                    # start searching for contacts
                    grasp, c1, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(graspable, x1, v, self.gripper.max_width,
                                                                                             min_grasp_width_world=self.gripper.min_width,

                                                                                             vis=vis)

                    if grasp is None or c2 is None:
                        continue

                    # get true contacts (previous is subject to variation)
                    success, c = grasp.find_contacts(graspable)
                    if not success:
                        continue
                    c1 = c[0]
                    c2 = c[1]

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        plt.figure()
                        ax = plt.gca(projection='3d')
                        c1_proxy = c1.plot_friction_cone(color='m')
                        c2_proxy = c2.plot_friction_cone(color='y')
                        ax.view_init(elev=5.0, azim=0)
                        plt.show(block=False)
                        time.sleep(0.5)
                        plt.close() # lol

                    # check friction cone
                    if PointGraspMetrics3D.force_closure(contacts=c,
                                                         params=params):
                        grasps.append(grasp)

        # randomly sample max num grasps from total list
        random.shuffle(grasps)
        return grasps

class MeshAntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone, then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """
    def sample_from_cone(self, B, n, tx, ty):
        """ Samples directoins from within the friction cone using uniform sampling.
        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector
        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
        """
        num_samples = n.shape[0]
        # theta = 2 * np.pi * np.random.rand(num_samples)
        theta = np.array([2 * np.pi * np.random.rand(num_samples) for i in range(B)]).reshape(B,-1)
        # r = self.friction_coef * np.abs(np.random.rand(num_samples))
        r = np.array([self.friction_coef * np.abs(np.random.rand(num_samples)) for i in range(B)]).reshape(B,-1)
        tx_mag = r * np.cos(theta)
        tx_mag = np.tile(tx_mag[:,:,np.newaxis], [1,1,3])
        ty_mag = r * np.sin(theta)
        ty_mag = np.tile(ty_mag[:,:,np.newaxis], [1,1,3])
        n = np.tile(n[np.newaxis,:,:], [B, 1, 1])
        tx = np.tile(tx[np.newaxis,:,:], [B, 1, 1])
        ty = np.tile(ty[np.newaxis,:,:], [B, 1, 1])
        v = n + tx_mag * tx + ty_mag * ty
        return v

    def multi_grasp(self, n, mesh, graspable, num_grasps, gripper=None):
        remain_grasp = copy.deepcopy(num_grasps)
        iter = 0
        grasps = []
        while remain_grasp > 0 and iter < 3:
            print("Thread %d remain %d grasps" % (n, remain_grasp))
            mesh_start = time.time()
            try:
                surface_points, face_index = ts.sample_surface_even(mesh, self.max_num_surface_points)
                print('Num surface: %d' %(len(surface_points)))
            except IndexError as ind:
                print(ind)
                return []
            print('Sample surface took %.3f sec' %(time.time() - mesh_start), "Thread %d" % n)


            # form proximity query structure
            proximity_start = time.time()
            surface_mesh = trimesh.Trimesh(vertices=surface_points)
            prox_query = tp.ProximityQuery(surface_mesh)
            print('Proximity took %.3f sec' %(time.time() - proximity_start), "Thread %d" % n)

            # compute closest faces to sample points
            faces_start = time.time()
            _, _, triangle_ids = tp.closest_point(mesh, surface_points)
            print('Faces took %.3f sec' %(time.time() - faces_start), "Thread %d" % n)

            # compute surface normals
            normals_start = time.time()
            surface_normals = mesh.face_normals[triangle_ids,:]
            print('Normals took %.3f sec' %(time.time() - normals_start), "Thread %d" % n)

            # compute candidate contact points
            intersection_start = time.time()
            grasp_dirs = -surface_normals
            flip_axis = 1 * (np.random.rand(triangle_ids.shape[0]) > 0.5)
            grasp_dirs[flip_axis==1] = -grasp_dirs[flip_axis==1]
            tx = np.c_[grasp_dirs[:,1], -grasp_dirs[:,0], np.zeros(grasp_dirs.shape[0])]
            tx[np.sum(tx, axis=1) == 0] = np.array([1,0,0])
            tx = tx / np.tile(np.linalg.norm(tx, axis=1)[:,np.newaxis], [1,3])
            ty = np.cross(grasp_dirs, tx)
            grasp_axes_list = self.sample_from_cone(1, grasp_dirs, tx, ty)

            grasp_axes = np.squeeze(np.array(grasp_axes_list), axis=0)
            locations1, ray_indices1, _ = mesh.ray.intersects_location(surface_points, grasp_axes)
            locations2, ray_indices2, _ = mesh.ray.intersects_location(surface_points, -grasp_axes)
            locations = np.r_[locations1, locations2]
            ray_indices = np.r_[ray_indices1, ray_indices2]
            print('Ray intersection took %.3f sec' %(time.time() - intersection_start), "Thread %d" % n)


            # find other contact point for a given direction
            nearest_start = time.time()
            distance, vertex_indices = prox_query.vertex(locations)
            print('Nearest took %.3f sec' %(time.time() - nearest_start), "Thread %d" % n)


            # index the arrays
            contacts_start = time.time()
            contact_points1 = surface_points[ray_indices,:]
            contact_points2 = surface_points[vertex_indices,:]
            contact_normals1 = surface_normals[ray_indices,:]
            contact_normals2 = surface_normals[vertex_indices,:]
            print('Contacts took %.3f sec' %(time.time() - contacts_start), "Thread %d" % n)

            # compute antipodality
            antipodal_start = time.time()
            v = contact_points1 - contact_points2
            v_norm = np.linalg.norm(v, axis=1)
            valid_indices = np.where((v_norm > 0) & (v_norm < self.gripper.max_width))[0]
            contact_points1 = contact_points1[valid_indices,:]
            contact_points2 = contact_points2[valid_indices,:]
            contact_normals1 = contact_normals1[valid_indices,:]
            contact_normals2 = contact_normals2[valid_indices,:]
            v = v[valid_indices,:]
            v_norm = v_norm[valid_indices]
            v = v / np.tile(v_norm[:,np.newaxis], [1,3])
            ip1 = np.abs(np.sum(contact_normals1 * v, axis=1))
            ip2 = np.abs(np.sum(contact_normals2 * (-v), axis=1))
            beta1 = np.arccos(ip1)
            beta2 = np.arccos(ip2)
            alpha = np.arctan(self.friction_coef)
            antipodal_ind = np.where((beta1 < alpha) & (beta2 < alpha))[0]
            print('Antipodal took %.3f sec' %(time.time() - antipodal_start), "Thread %d" % n)


            # form list of grasp objects
            grasp_start = time.time()
            grasp_points1 = contact_points1[antipodal_ind,:]
            grasp_points2 = contact_points2[antipodal_ind,:]
            grasp_normals1 = contact_normals1[antipodal_ind, :]
            grasp_normals2 = contact_normals2[antipodal_ind, :]
            total_grasps = grasp_points1.shape[0]
            grasp_centers = 0.5 * (grasp_points1 + grasp_points2)
            grasp_axes = v[antipodal_ind,:]
            grasp_widths = self.gripper.max_width * np.ones(grasp_axes.shape[0])
            ind = np.arange(total_grasps)
            np.random.shuffle(ind)

            k = 0
            while len(grasps) < num_grasps and k < total_grasps:
                # create grasp object
                i = ind[k]
                if (grasp_points1[i]==grasp_points2[i]).all():
                    continue
                for theta in [0,1,2,3]:
                    grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i,:], grasp_axes[i,:], grasp_widths[i], grasp_points1[i], grasp_points2[i], grasp_normals1[i], grasp_normals2[i], angle=theta*np.pi/2))

                # check collisions
                    if self.check_collisions:

                        is_collision = self.collides_along_approach(n, copy.deepcopy(grasp), graspable, gripper=gripper,
                                                                approach_dist=self.approach_dist,
                                                                delta_approach=self.delta_approach)
                        if not is_collision:
                            grasps.append(copy.deepcopy(grasp))
                    # trimesh.Scene([create_robotiq_marker(color=[0, 255, 255]).apply_transform((grasp.gripper_pose(gripper) * gripper.T_mesh_gripper.inverse()).matrix)] + [graspable.mesh.trimesh]).show()


                k += 1
            print('Grasps %d/%d took %.3f sec' %(k, num_grasps, time.time() - grasp_start), "Thread %d" % n)
            remain_grasp = copy.deepcopy(num_grasps) - len(grasps)
            print('After iter %d For %d grasps, remain %d' %(iter, num_grasps, remain_grasp), "Thread %d" % n)
            iter += 1
        return grasps


    def sample_hierachical_grasps(self, graspable, num_grasps, gripper=None,vis=False):
        def print_error(value):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ")
            print(value)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ")

        mesh = graspable.mesh.trimesh

        cell_bounds = np.linspace(mesh.bounds[0, :]-0.1, mesh.bounds[1, :]+0.1, 3)
        diag_center = np.zeros((len(cell_bounds)-1,3))
        cell_centers = []
        for i in range(len(cell_bounds)-1):
            diag_center[i] = (cell_bounds[i+1, :] + cell_bounds[i, :]) / 2
        for x in diag_center[:, 0]:
            for y in diag_center[:, 1]:
                for z in diag_center[:, 2]:
                    cell_centers.append(np.array([x,y,z]))

        submesh_list = []
        bbox_list = []

        # block antipodal sampling
        for cell_center in cell_centers:
            bbox = trimesh.creation.box(extents=(cell_bounds[1, :] - cell_bounds[0, :] + 2 * self.gripper.max_width), transform=RigidTransform(np.eye(3), cell_center).matrix)
            submesh = mesh.slice_plane(bbox.facets_origin, -bbox.facets_normal)
            bbox_list.append(bbox)
            submesh_list.append(submesh)

        # trimesh.Scene(bbox_list+[mesh]).show()
        # trimesh.Scene(submesh_list[1]).show()


        grasp_list = []

        # for i in range(len(submesh_list)):
        #     grasps0 = self.multi_grasp(i, submesh_list[i], graspable, math.ceil(float(num_grasps / len(submesh_list))), gripper)
        #     grippers = []
        #     for grasp in grasps0:
        #         grippers.append(create_robotiq_marker(color=[0, 255, 255]).apply_transform((grasp.gripper_pose(self.gripper) * self.gripper.T_mesh_gripper.inverse()).matrix))
        #     trimesh.Scene(grippers + [submesh_list[i]]).show()

        t0 = time.time()
        subgrasp = math.ceil(float(num_grasps / len(submesh_list)))
        for i in range(len(submesh_list)):
            grasp_list.append(self.multi_grasp(i, submesh_list[i], graspable, subgrasp, gripper))

        grasp_candidate = []

        for grasp in grasp_list:
            grasp_candidate = np.concatenate((grasp_candidate, np.array(grasp)), axis=0)


        print("All cost: %.3f sec" % (time.time() - t0))

        return grasp_candidate


    def collides_along_approach(self, n, grasp, graspable, gripper, approach_dist, delta_approach):
        """ Checks whether a grasp collides along its approach direction.
        Currently assumes that the collision checker has loaded the object.
        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp to check collisions for
        gripper: 'RobotGripper'
        approach_dist : float
            how far back to check along the approach direction
        delta_approach : float
            how finely to discretize poses along the approach direction
        key : str
            key of object to grasp
        Returns
        -------
        bool
            whether or not the grasp is in collision
        """
        collision_checker = tc.CollisionManager()
        grasp_dict = {}
        i = 0
        # get the gripper pose and axis of approach
        T_obj_grasp= grasp.T_obj_grasp
        grasp_approach_axis = T_obj_grasp.x_axis

        # setup variables

        cur_approach = 0.0
        while cur_approach <= approach_dist:
            # back up along approach dir
            T_obj_approach = T_obj_grasp.copy()
            T_obj_approach.translation -= cur_approach * grasp_approach_axis
            T_obj_gripper = T_obj_approach * gripper.T_grasp_gripper
            T_obj_mesh = T_obj_gripper * gripper.T_mesh_gripper.inverse()
            grasp.T_obj_mesh = T_obj_mesh
            grasp_dict[str(i)] = copy.copy(grasp)

            # check collisions
            collision_checker.add_object(str(i), create_robotiq_marker(), T_obj_mesh.matrix)
            cur_approach += delta_approach
            i += 1
        if isinstance(graspable, GraspableObject3D):
            is_collision, grasp_names = collision_checker.in_collision_single(graspable.mesh.trimesh, return_names=True)
        else:
            is_collision, grasp_names = collision_checker.in_collision_single(graspable, return_names=True)
        scale = len(grasp_dict)
        for name in grasp_names:
            grasp_dict.pop(name)
        # if n == 3 and not is_collision:
        #     for name, success in grasp_dict.items():
        #         grippers.append(create_robotiq_marker(color=[0, int(name)/scale*255, 0]).apply_transform(success.T_obj_mesh.matrix)) # please note that this exists a deepcopy problem! We must create marker every single time.
        #     trimesh.Scene(grippers + [graspable.mesh.trimesh]).show()
        return is_collision





class EdgeAvoidanceMeshAntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone, then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """
    def sample_from_cone(self, n, tx, ty):
        """ Samples directoins from within the friction cone using uniform sampling.

        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector

        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
        """
        num_samples = n.shape[0]
        theta = 2 * np.pi * np.random.rand(num_samples)
        r = self.friction_coef * np.abs(np.random.rand(num_samples))
        tx_mag = r * np.cos(theta)
        tx_mag = np.tile(tx_mag[:,np.newaxis], [1,3])
        ty_mag = r * np.sin(theta)
        ty_mag = np.tile(ty_mag[:,np.newaxis], [1,3])
        v = n + tx_mag * tx + ty_mag * ty
        return v

    def sample_grasps(self, graspable, num_grasps,
                      vis=False):
        """Returns a list of candidate grasps for graspable object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            number of grasps to sample
        vis : bool
            whether or not to visualize progress, for debugging

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            the sampled grasps
        """
        # get all surface points
        mesh_start = time.time()
        mesh = graspable.mesh.trimesh
        surface_points = ts.sample_surface_even(mesh, self.max_num_surface_points)
        logging.debug('Sample surface took %.3f sec' %(time.time() - mesh_start))
        logging.debug('Num surface: %d' %(len(surface_points)))

        # compute closest faces to sample points
        faces_start = time.time()
        _, _, triangle_ids = tp.closest_point(mesh, surface_points)
        logging.debug('Faces took %.3f sec' %(time.time() - faces_start))

        # form proximity query structure
        proximity_start = time.time()
        surface_mesh = trimesh.Trimesh(vertices=surface_points)
        prox_query = tp.ProximityQuery(surface_mesh)
        logging.debug('Proximity took %.3f sec' %(time.time() - proximity_start))

        # compute surface normals
        normals_start = time.time()
        surface_normals = mesh.face_normals[triangle_ids,:]
        logging.debug('Normals took %.3f sec' %(time.time() - normals_start))

        # compute candidate contact points
        intersection_start = time.time()
        grasp_dirs = -surface_normals
        flip_axis = 1 * (np.random.rand(triangle_ids.shape[0]) > 0.5)
        grasp_dirs[flip_axis==1] = -grasp_dirs[flip_axis==1]
        tx = np.c_[grasp_dirs[:,1], -grasp_dirs[:,0], np.zeros(grasp_dirs.shape[0])]
        tx[np.sum(tx, axis=1) == 0] = np.array([1,0,0])
        tx = tx / np.tile(np.linalg.norm(tx, axis=1)[:,np.newaxis], [1,3])
        ty = np.cross(grasp_dirs, tx)
        grasp_axes = self.sample_from_cone(grasp_dirs, tx, ty)

        locations1, ray_indices1, _ = mesh.ray.intersects_location(surface_points, grasp_axes)
        locations2, ray_indices2, _ = mesh.ray.intersects_location(surface_points, -grasp_axes)
        locations = np.r_[locations1, locations2]
        ray_indices = np.r_[ray_indices1, ray_indices2]
        logging.debug('Ray intersection took %.3f sec' %(time.time() - intersection_start))

        # find other contact point for a given direction
        nearest_start = time.time()
        _, vertex_indices = prox_query.vertex(locations)
        logging.debug('Nearest took %.3f sec' %(time.time() - nearest_start))

        # index the arrays
        contacts_start = time.time()
        contact_points1 = surface_points[ray_indices,:]
        contact_points2 = surface_points[vertex_indices,:]
        contact_normals1 = surface_normals[ray_indices,:]
        contact_normals2 = surface_normals[vertex_indices,:]
        logging.debug('Contacts took %.3f sec' %(time.time() - contacts_start))

        # compute antipodality
        antipodal_start = time.time()
        v = contact_points1 - contact_points2
        v_norm = np.linalg.norm(v, axis=1)
        valid_indices = np.where((v_norm > 0) & (v_norm < self.gripper.max_width))[0]
        contact_points1 = contact_points1[valid_indices,:]
        contact_points2 = contact_points2[valid_indices,:]
        contact_normals1 = contact_normals1[valid_indices,:]
        contact_normals2 = contact_normals2[valid_indices,:]
        v = v[valid_indices,:]
        v_norm = v_norm[valid_indices]
        v = v / np.tile(v_norm[:,np.newaxis], [1,3])
        ip1 = np.abs(np.sum(contact_normals1 * v, axis=1))
        ip2 = np.abs(np.sum(contact_normals2 * (-v), axis=1))
        beta1 = np.arccos(ip1)
        beta2 = np.arccos(ip2)
        alpha = np.arctan(self.friction_coef)
        antipodal_ind = np.where((beta1 < alpha) & (beta2 < alpha))[0]
        logging.debug('Antipodal took %.3f sec' %(time.time() - antipodal_start))

        # form list of grasp objects
        grasp_start = time.time()
        grasp_points1 = contact_points1[antipodal_ind,:]
        grasp_points2 = contact_points2[antipodal_ind,:]

        # Reject grasps too close to edges with sharp angles -- wip code
        if self.sampling_reject_use_extra_checking:
            angles = mesh.face_adjacency_angles
            edges_idx = np.where(angles > self.sampling_reject_angle_threshold)
            vertices_idx = mesh.face_adjacency_edges[edges_idx]
            edges = mesh.vertices[vertices_idx]

            accepted_grasps = []
            for ep1, ep2 in zip(grasp_points1, grasp_points2):
                accepted_single = True
                for ep in (ep1, ep2):
                    dists = np.linalg.norm(np.cross(edges[:,0] - edges[:,1], ep - edges[:,1], axis=1), axis=1) / np.linalg.norm(edges[:,0] - edges[:,1], axis=1)
                    if np.any(np.abs(dists) < self.sampling_reject_edge_distance):
                        accepted_single = False
                        break
                accepted_grasps.append(accepted_single)
        else:
            # Reject grasps too close to edges with sharp angles
            # vectorize this further later for speedup
            _, _, triangle_ids_g1 = tp.closest_point(mesh, grasp_points1)
            _, _, triangle_ids_g2 = tp.closest_point(mesh, grasp_points2)
            accepted_grasps = []
            for grasp_endpoint_1, grasp_endpoint_2, triangle_id_1, triangle_id_2 in zip(grasp_points1, grasp_points2, triangle_ids_g1, triangle_ids_g2):
                accepted_single = True
                for grasp_endpoint, triangle_id in zip([grasp_endpoint_1, grasp_endpoint_2],[triangle_id_1, triangle_id_2]):
                    edges_idx = np.where(np.bitwise_or(mesh.face_adjacency[:, 0] == triangle_id, mesh.face_adjacency[:, 1] == triangle_id))
                    vertices_idx = mesh.face_adjacency_edges[edges_idx]
                    edges = mesh.vertices[vertices_idx]
                    angles = mesh.face_adjacency_angles[edges_idx]

                    for (v1, v2), angle in zip(edges, angles):
                        if (angle > self.sampling_reject_angle_threshold):
                            dist = np.linalg.norm(np.cross(v2 - v1, grasp_endpoint - v1)) / np.linalg.norm(v2 -v1)
                            if np.abs(dist) < self.sampling_reject_edge_distance:
                                accepted_single = False
                                break
                    if accepted_single == False:
                        break
                #if accepted_single == True:
                #    np.set_printoptions(threshold=np.nan)
                #    raise Exception("{}\n{}\n{}\n{}\n{}\n{}\n".format(grasp_endpoint_1, grasp_endpoint_2, triangle_id_1, triangle_id_2, grasp_points1, grasp_points2))
                accepted_grasps.append(accepted_single)

        accepted_grasps = np.asarray(accepted_grasps)
        grasp_points1 = grasp_points1[accepted_grasps]
        grasp_points2 = grasp_points2[accepted_grasps]

        grasp_axes = v[antipodal_ind,:][accepted_grasps]

        total_grasps = grasp_points1.shape[0]
        grasp_centers = 0.5 * (grasp_points1 + grasp_points2)
        grasp_widths = self.gripper.max_width * np.ones(grasp_axes.shape[0])
        ind = np.arange(total_grasps)
        np.random.shuffle(ind)

        if self.check_collisions:
            collision_checker = GraspCollisionChecker(self.gripper)
            collision_checker.set_graspable_object(graspable)

        grasps = []
        k = 0
        while len(grasps) < num_grasps and k < total_grasps:
            # create grasp object
            i = ind[k]
            grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i,:], grasp_axes[i,:], grasp_widths[i]))

            # check collisions
            collision_free = True
            if self.check_collisions:
                collision_free = False
                for j in range(self.num_grasp_rots):
                    theta = 2 * np.pi * j / self.num_grasp_rots
                    grasp.approach_angle = theta
                    collides = collision_checker.collides_along_approach(grasp, self.approach_dist, self.delta_approach)
                    if not collides:
                        collision_free = True
                        break

            # append to list
            if collision_free:
                grasps.append(grasp)
            k += 1

        logging.debug('Grasps %d took %.3f sec' %(num_grasps, time.time() - grasp_start))
        return grasps