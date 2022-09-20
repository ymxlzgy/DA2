"""
Quasi-static point-based grasp quality metrics.
Author: Jeff Mahler and Brian Hou
"""
import logging
import torch
import numpy as np
try:
    import pyhull.convex_hull as cvh
except:
    logging.warning('Failed to import pyhull')
try:
    import cvxopt as cvx
except:
    logging.warning('Failed to import cvx')
import os
import scipy.spatial as ss
import sys
import time

from dexnet.grasping import PointGrasp, GraspableObject3D, GraspQualityConfig
from dexnet.grasping import FCLoss

import meshpy.obj_file as obj_file
import meshpy.sdf_file as sdf_file

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
import IPython

# turn off output logging
cvx.solvers.options['show_progress'] = False

class PointGraspMetrics3D:
    """ Class to wrap functions for quasistatic point grasp quality metrics.
    """

    @staticmethod
    def grasp_quality(grasp, obj, params, vis = False):
        """
        Computes the quality of a two-finger point grasps on a given object using a quasi-static model.

        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp to evaluate
        obj : :obj:`GraspableObject3D`
            object to evaluate quality on
        params : :obj:`GraspQualityConfig`
            parameters of grasp quality function
        """
        start = time.time()
        if not isinstance(grasp, PointGrasp):
            raise ValueError('Must provide a point grasp object')
        if not isinstance(obj, GraspableObject3D):
            raise ValueError('Must provide a 3D graspable object')
        if not isinstance(params, GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')

        # read in params
        method = params.quality_method
        friction_coef = params.friction_coef
        num_cone_faces = params.num_cone_faces
        soft_fingers = params.soft_fingers
        check_approach = params.check_approach
        if not hasattr(PointGraspMetrics3D, method):
            raise ValueError('Illegal point grasp metric %s specified' %(method))

        # get point grasp contacts
        contacts_start = time.time()
        contacts_found, contacts = grasp.find_contacts(obj,
                                                       check_approach=check_approach,
                                                       vis=vis,
                                                       params=params)
        if not contacts_found:
            logging.debug('Contacts not found')
            return 0

        # add the forces, torques, etc at each contact point
        forces_start = time.time()
        num_contacts = len(contacts)
        forces = np.zeros([3,0])
        torques = np.zeros([3,0])
        normals = np.zeros([3,0])
        for i in range(num_contacts):
            contact = contacts[i]
            if vis:
                if i == 0:
                    contact.plot_friction_cone(color='y')
                else:
                    contact.plot_friction_cone(color='c')

            # get contact forces
            force_success, contact_forces, contact_outward_normal = contact.friction_cone(num_cone_faces, friction_coef)

            if not force_success:
                logging.debug('Force computation failed')
                if params.all_contacts_required:
                    return 0

            # get contact torques
            torque_success, contact_torques = contact.torques(contact_forces)
            if not torque_success:
                logging.debug('Torque computation failed')
                if params.all_contacts_required:
                    return 0

            # get the magnitude of the normal force that the contacts could apply
            n = contact.normal_force_magnitude()

            forces = np.c_[forces, n * contact_forces]
            torques = np.c_[torques, n * contact_torques]
            normals = np.c_[normals, n * -contact_outward_normal] # store inward pointing normals

        if normals.shape[1] == 0:
            logging.debug('No normals')
            return 0

        # normalize torques
        if 'torque_scaling' not in params.keys():
            params.torque_scaling = 1.0

        if vis:
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()

        # evaluate the desired quality metric
        quality_start = time.time()
        Q_func = getattr(PointGraspMetrics3D, method)
        quality = Q_func(forces, torques, normals,
                         soft_fingers=soft_fingers,
                         contacts=contacts,
                         obj=obj,
                         params=params)

        end = time.time()
        logging.debug('Contacts took %.3f sec' %(forces_start - contacts_start))
        logging.debug('Forces took %.3f sec' %(quality_start - forces_start))
        logging.debug('Quality eval took %.3f sec' %(end - quality_start))
        logging.debug('Everything took %.3f sec' %(end - start))

        return quality

    @staticmethod
    def grasp_matrix(forces, torques, normals, soft_fingers=False,
                     finger_radius=0.005, params=None):
        """ Computes the grasp map between contact forces and wrenchs on the object in its reference frame.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        finger_radius : float
            the radius of the fingers to use
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        G : 6xM :obj:`numpy.ndarray`
            grasp map
        """
        if params is not None:
            if 'finger_radius' in params.keys():
                finger_radius = params.finger_radius

        num_forces = forces.shape[1]
        num_torques = torques.shape[1]
        if num_forces != num_torques:
            raise ValueError('Need same number of forces and torques')

        num_cols = num_forces
        if soft_fingers:
            num_normals = 2
            if normals.ndim > 1:
                num_normals = 2*normals.shape[1]
            num_cols = num_cols + num_normals

        G = np.zeros([6, num_cols])
        for i in range(num_forces):
            G[:3,i] = forces[:,i]
            G[3:,i] = params.torque_scaling * torques[:,i]

        if soft_fingers:
            torsion = np.pi * finger_radius**2 * params.friction_coef * normals * params.torque_scaling
            pos_normal_i = -num_normals
            neg_normal_i = -num_normals + num_normals / 2
            G[3:,pos_normal_i:neg_normal_i] = torsion
            G[3:,neg_normal_i:] = -torsion

        return G

    @staticmethod
    def force_closure(forces=None, torques=None, normals=None, soft_fingers=False,
                      contacts=None, obj=None,
                      wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                      params=None):
        """" Checks force closure using the antipodality trick.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        # read params
        c1 = contacts[0]
        c2 = contacts[1]
        friction_coef = params['friction_coef']

        use_abs_value = True
        if 'use_abs_value' in params.keys():
            use_abs_value = params['use_abs_value']

        if c1.point is None or c2.point is None or c1.normal is None or c2.normal is None:
            return 0
        p1, p2 = c1.point, c2.point
        n1, n2 = -c1.normal, -c2.normal # inward facing normals

        if (p1 == p2).all(): # same point
            return 0

        for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
            diff = other_contact - contact
            normal_proj = normal.dot(diff) / np.linalg.norm(normal)
            if use_abs_value:
                normal_proj = abs(normal.dot(diff)) / np.linalg.norm(normal)

            if normal_proj < 0:
                return 0 # wrong side
            alpha = np.arccos(normal_proj / np.linalg.norm(diff))
            if alpha > np.arctan(friction_coef):
                return 0 # outside of friction cone
        return 1

    @staticmethod
    def G_v(points):
        """"
        Parameters
        ----------
        points : BxNx3 :obj:`numpy.ndarray`
            surface points at the contact points
        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        fc_loss = FCLoss()
        points = torch.tensor(points).cuda()
        G_f = fc_loss.x_to_G(points).cpu().numpy()
        G_f_extended = np.c_[G_f[:,:,0:3], np.zeros((len(G_f), 6, 3))]
        for i in range(1,4):
            G_f_extended = np.dstack((G_f_extended, np.c_[G_f[:,:,i*3:i*3+3], np.zeros((len(G_f), 6, 3))]))
        G_v = np.matmul(np.linalg.inv(np.matmul(G_f_extended, G_f_extended.transpose((0,2,1)))), G_f_extended)

        return G_f_extended, G_v

    # @staticmethod
    # def Volume_ellipsoid(contacts1=None, contacts2=None):
    #     fc_loss = FCLoss()
    #     c1 = contacts1[0]
    #     c2 = contacts1[1]
    #     c3 = contacts2[0]
    #     c4 = contacts2[1]
    #     p1, p2, p3, p4 = c1.point, c2.point, c3.point, c4.point
    #     points = np.c_[p1, p2, p3, p4].T
    #     points = torch.tensor(points[np.newaxis, :]).cuda()
    #     G_f = np.squeeze(fc_loss.x_to_G(points).cpu().numpy(),0)
    #     G_v = np.matmul(np.linalg.inv(np.matmul(G_f, G_f.T)), G_f)
    #     w = np.sqrt(np.linalg.det(np.matmul(G_v, G_v.T)))
    #     return w

    @staticmethod
    def Dexterity(contacts):
        G_f, G_v = PointGraspMetrics3D.G_v(contacts)
        sigma = np.linalg.svd(G_v, compute_uv=False)
        return np.min(sigma, axis=1)

    @staticmethod
    def Torque_optimization(contacts):
        G_f, G_v = PointGraspMetrics3D.G_v(contacts)
        c = np.linalg.eig(np.matmul(G_v, G_v.transpose((0,2,1))))
        v = c[1][np.arange(len(c[1])),np.argmin(c[0], axis=1),:]
        sim1 = 0.5 + 0.5 * (v[:,2] / np.linalg.norm(v[:,0:3], axis=1))
        # sim2 =
        return sim1

    @staticmethod
    def Dual_force_closure_batch(points, normals, params=None):
        """" Checks force closure using the antipodality trick.

        Parameters
        ----------
        points : BxNx3 :obj:`numpy.ndarray`
            surface points at the contact points
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        fc_loss = FCLoss()
        points = torch.tensor(points).cuda()
        normals = torch.tensor(normals).cuda()
        result1, result2 = fc_loss.fc_loss(points, normals)
        result1 = result1.cpu().numpy()
        result2 = result2.cpu().numpy()

        result1 = np.where(result1 < 0.01, result1, 0) # 0 is not qualified
        result2 = np.where(result2 < 1, result2, 0) # 0 is not qualified
        print("result1: ", result1)
        print("result2: ", result2)



        return  result1 * result2, result2

    @staticmethod
    def Dual_force_closure(forces=None, torques=None, normals=None, soft_fingers=False,
                      contacts1=None, contacts2=None, obj=None,
                      wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                      params=None):
        """" Checks force closure using the antipodality trick.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        fc_loss = FCLoss()
        # read params
        c1 = contacts1[0]
        c2 = contacts1[1]
        c3 = contacts2[0]
        c4 = contacts2[1]
        friction_coef = params['friction_coef']

        use_abs_value = True
        if 'use_abs_value' in params.keys():
            use_abs_value = params['use_abs_value']

        if c1.point is None or c2.point is None or c1.normal is None or c2.normal is None or c3.point is None or c4.point is None or c3.normal is None or c4.normal is None:
            return 0
        p1, p2, p3, p4 = c1.point, c2.point, c3.point, c4.point
        n1, n2, n3, n4 = -c1.normal, -c2.normal, -c3.normal, -c4.normal # inward facing normals

        if (p1 == p2).all(): # same point
            return 0
        elif (p3 == p4).all():
            return 0
        points = np.c_[p1, p2, p3, p4].T
        points = torch.tensor(points[np.newaxis, :]).cuda()
        normals = np.c_[n1, n2, n3, n4].T
        normals = torch.tensor(normals[np.newaxis, :]).cuda()
        result1, result2 = fc_loss.fc_loss(points, normals)
        if result1 > 0.01 or result2 > 0.5:
            return 0

        return 1

    @staticmethod
    def force_closure_qp(forces, torques, normals, soft_fingers=False,
                         contacts=None, obj=None,
                         wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                         params=None):
        """ Checks force closure by solving a quadratic program (whether or not zero is in the convex hull)

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        if params is not None:
            if 'wrench_norm_thresh' in params.keys():
                wrench_norm_thresh = params.wrench_norm_thresh
            if 'wrench_regularizer' in params.keys():
                wrench_regularizer = params.wrench_regularizer

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers, params=params)
        min_norm, _ = PointGraspMetrics3D.min_norm_vector_in_facet(G, wrench_regularizer=wrench_regularizer)
        return 1 * (min_norm < wrench_norm_thresh) # if greater than wrench_norm_thresh, 0 is outside of hull

    @staticmethod
    def max_contact_pct_incr(forces, torques, normals, soft_fingers=False,
                             contacts=None, obj=None,
                             wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                             params=None):
        """ Computes the maximum spring percent increase for a suction contact.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float
            A value in [0,1] that represents the quality of the suction contact
            based on spring stretches
        """
        if params is None:
            raise ValueError('Cannot compute max_contact_pct_incr without contacts and object')

        # read params
        obj = params['obj']
        contacts = params['contacts']
        contact = contacts[0]

        if not isinstance(contact, SuctionContact3D):
            raise ValueError('Contact percent increase is only available for suction contacts!')
        if contact.struct_pct_incr is None or contact.flex_pct_incr is None or contact.cone_pct_incr is None:
            return 0

        max_ratio = max(contact.struct_pct_incr / params.max_struct_pct_incr,
                        contact.flex_pct_incr / params.max_flex_pct_incr,
                        contact.cone_pct_incr / params.max_cone_pct_incr)

        return np.exp(-1*max_ratio)

    @staticmethod
    def planarity_com(forces, torques, normals, soft_fingers=False,
                      contacts=None, obj=None,
                      wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                      params=None):
        """ Computes the inverse distance to the object center of mass
        for all contacts that are sufficiently planar.
        (e.g. SSE from the approach plane is less than a threshold value)

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float
            If the sum of squared errors from the approach plane is less than a threshold,
            then returns exp(-d) where d is the center of mass. Otherwise returns zero.
        """
        if params is None:
            raise ValueError('Cannot compute max_contact_pct_incr without contacts and object')

        # read params
        obj = params['obj']
        contacts = params['contacts']
        contact = contacts[0]

        if not isinstance(contact, SuctionContact3D):
            raise ValueError('Planarity COM is only available for suction contacts!')
        if contact.struct_pct_incr is None or contact.flex_pct_incr is None or contact.cone_pct_incr is None:
            return 0

        # compute patch planarity
        p = contact.planarity(params.num_samples)
        if p > params.planarity_thresh:
            return 0

        # compute distance to center of mass
        d = np.linalg.norm(obj.moment_arm(contact.point))

        # scale by max moment arm
        _, max_coords = obj.mesh.bounding_box()
        rho = 2*np.max(max_coords)

        return (np.exp(-d / rho) - np.exp(-1)) / (1 - np.exp(-1))

    @staticmethod
    def partial_closure(forces, torques, normals, soft_fingers=False,
                        contacts=None, obj=None,
                        wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                        params=None):
        """ Evalutes partial closure: whether or not the forces and torques can resist a specific wrench.
        Estimates resistance by sollving a quadratic program (whether or not the target wrench is in the convex hull).

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in partial closure, 0 otherwise
        """
        force_limit = None
        if params is None:
            return 0
        force_limit = params.force_limits
        target_wrench = params.target_wrench
        target_wrench[3:] = params.torque_scaling * target_wrench[3:] # set torque scaling
        if 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer

        # reorganize the grasp matrix for easier constraint enforcement in optimization
        num_fingers = normals.shape[1]
        num_wrenches_per_finger = forces.shape[1] / num_fingers
        G = np.zeros([6,0])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            G_i = PointGraspMetrics3D.grasp_matrix(forces[:,start_i:end_i], torques[:,start_i:end_i], normals[:,i:i+1],
                                                   soft_fingers, params=params)
            G = np.c_[G, G_i]


        wrench_resisted, _ = PointGraspMetrics3D.wrench_in_positive_span(G, target_wrench, force_limit, num_fingers,
                                                                         wrench_norm_thresh=wrench_norm_thresh,
                                                                         wrench_regularizer=wrench_regularizer)
        return 1 * wrench_resisted

    @staticmethod
    def wrench_resistance(forces, torques, normals, soft_fingers=False,
                          contacts=None, obj=None,
                          wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                          finger_force_eps=1e-9, params=None):
        """ Evalutes wrench resistance: the inverse norm of the contact forces required to resist a target wrench
        Estimates resistance by solving a quadratic program (min normal contact forces to produce a wrench).

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        finger_force_eps : float
            small float to prevent numeric issues in wrench resistance metric
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of wrench resistance metric
        """
        force_limit = None
        if params is None:
            return 0
        force_limit = params.force_limits
        target_wrench = params.target_wrench
        target_wrench[3:] = params.torque_scaling * target_wrench[3:]  # set torque scaling
        if 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer
        if 'finger_force_eps' in params.keys():
            finger_force_eps = params.finger_force_eps

        # reorganize the grasp matrix for easier constraint enforcement in optimization
        num_fingers = normals.shape[1]
        num_wrenches_per_finger = forces.shape[1] / num_fingers
        G = np.zeros([6,0])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            G_i = PointGraspMetrics3D.grasp_matrix(forces[:,start_i:end_i], torques[:,start_i:end_i], normals[:,i:i+1],
                                                   soft_fingers, params=params)
            G = np.c_[G, G_i]

        # compute metric from finger force norm
        Q = 0
        wrench_resisted, alpha = PointGraspMetrics3D.wrench_in_positive_span(G, target_wrench, force_limit, num_fingers,
                                                                             wrench_norm_thresh=wrench_norm_thresh,
                                                                             wrench_regularizer=wrench_regularizer)
        if wrench_resisted:
            Q = np.linalg.norm(target_wrench) / np.linalg.norm(alpha)
        return Q

    @staticmethod
    def suction_wrench_resistance(forces, torques, normals, soft_fingers=False,
                                  contacts=None, obj=None,
                                  wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                                  finger_force_eps=1e-9, params=None):
        """ Evalutes wrench resistance: the inverse norm of the contact forces required to resist a target wrench
        Estimates resistance by sollving a quadratic program (min normal contact forces to produce a wrench).

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        finger_force_eps : float
            small float to prevent numeric issues in wrench resistance metric
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of wrench resistance metric
        """
        # read params
        force_limit = None
        if params is None:
            return 0
        force_limit = params.force_limits
        if 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer
        if 'finger_force_eps' in params.keys():
            finger_force_eps = params.finger_force_eps

        # read contacts
        contacts = params['contacts']
        contact = contacts[0]

        # set grasp matrix
        num_wrenches = 6
        normal, t1, t2 = contact.tangents()
        wrench_basis = np.zeros([num_wrenches,num_wrenches])
        wrench_basis[:,:3] = np.r_[forces, torques]
        wrench_basis[3:,3] = normal
        wrench_basis[3:,4] = t1
        wrench_basis[3:,5] = t2

        # set target wrench
        target_wrench = params.target_wrench.copy()
        target_wrench[3:] = params.torque_scaling * target_wrench[3:]  # set torque scaling

        # quadratic and linear costs
        P = wrench_basis.T.dot(wrench_basis) + wrench_regularizer*np.eye(num_wrenches)
        q = -wrench_basis.T.dot(target_wrench)

        # inequalities
        G = np.zeros([2*num_wrenches-1, num_wrenches])
        G[0,0] = 1
        G[0,2] = -(np.sqrt(3) / 3) * params.friction_coef

        G[1,1] = 1
        G[1,2] = -(np.sqrt(3) / 3) * params.friction_coef

        G[2,3] = 1
        G[2,2] = -(np.sqrt(3) / 3) * params.finger_radius * params.friction_coef * params.torque_scaling

        G[3,4] = 1
        G[4,5] = 1

        G[5,0] = -1
        G[5,2] = -(np.sqrt(3) / 3) * params.friction_coef

        G[6,1] = -1
        G[6,2] = -(np.sqrt(3) / 3) * params.friction_coef

        G[7,3] = -1
        G[7,2] = -(np.sqrt(3) / 3) * params.finger_radius * params.friction_coef * params.torque_scaling

        G[8,4] = -1
        G[9,5] = -1

        G[10,2] = -1

        h = np.zeros(2*num_wrenches-1)
        h[0] = (np.sqrt(3) / 3) * params.friction_coef * params.vacuum_force
        h[1] = (np.sqrt(3) / 3) * params.friction_coef * params.vacuum_force
        h[2] = (np.sqrt(3) / 3) * params.finger_radius * params.friction_coef * params.vacuum_force * params.torque_scaling
        h[3] = (np.sqrt(2) / 2) * np.pi * params.finger_radius * params.material_limit * params.torque_scaling
        h[4] = (np.sqrt(2) / 2) * np.pi * params.finger_radius * params.material_limit * params.torque_scaling
        h[5:10] = h[:5]
        h[10] = params.vacuum_force

        # convert to cvx and solve
        P = cvx.matrix(P)
        q = cvx.matrix(q)
        A = cvx.matrix(G)
        b = cvx.matrix(h)
        sol = cvx.solvers.qp(P, q, A, b)
        alpha = np.array(sol['x'])

        min_dist = np.linalg.norm(wrench_basis.dot(alpha).ravel() - target_wrench)**2

        # check whether or not the wrench was resisted
        Q = 0
        if min_dist < wrench_norm_thresh:
            Q = 1.0 #np.linalg.norm(target_wrench) / np.linalg.norm(alpha)
        return Q

    @staticmethod
    def min_singular(forces, torques, normals, soft_fingers=False,
                     contacts=None, obj=None,
                     params=None):
        """ Min singular value of grasp matrix - measure of wrench that grasp is "weakest" at resisting.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of smallest singular value
        """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers,
                                             params=params)
        _, S, _ = np.linalg.svd(G)
        min_sig = S[5]
        return min_sig

    @staticmethod
    def wrench_volume(forces, torques, normals, soft_fingers=False,
                      contacts=None, obj=None,
                      params=None):
        """ Volume of grasp matrix singular values - score of all wrenches that the grasp can resist.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of wrench volume
        """
        k = 1
        if params is not None and 'k' in params.keys():
            k = params.k

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        sig = S
        return k * np.sqrt(np.prod(sig))

    @staticmethod
    def grasp_isotropy(forces, torques, normals, soft_fingers=False, params=None):
        """ Condition number of grasp matrix - ratio of "weakest" wrench that the grasp can exert to the "strongest" one.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of grasp isotropy metric
        """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        max_sig = S[0]
        min_sig = S[5]
        isotropy = min_sig / max_sig
        if np.isnan(isotropy) or np.isinf(isotropy):
            return 0
        return isotropy

    @staticmethod
    def ferrari_canny_L1(forces, torques, normals, soft_fingers=False,
                         contacts=None, obj=None,
                         params=None,
                         wrench_norm_thresh=1e-3,
                         wrench_regularizer=1e-10):
        """ Ferrari & Canny's L1 metric. Also known as the epsilon metric.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float : value of metric
        """
        if params is not None and 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if params is not None and 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer

        # create grasp matrix
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals,
                                             soft_fingers, params=params)
        s = time.time()
        # center grasp matrix for better convex hull comp
        hull = cvh.ConvexHull(G.T)
        # TODO: suppress ridiculous amount of output for perfectly valid input to qhull
        e = time.time()
        logging.debug('CVH took %.3f sec' %(e - s))

        debug = False
        if debug:
            forces = G[:3,:].T
            torques = G[3:,:].T
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(forces[:,0], forces[:,1], forces[:,2], c='r', s=50)
            ax.scatter(0, 0, 0, c='k', s=80)
            ax.set_xlim3d(-1.5, 1.5)
            ax.set_ylim3d(-1.5, 1.5)
            ax.set_zlim3d(-1.5, 1.5)
            ax.set_xlabel('fx')
            ax.set_ylabel('fy')
            ax.set_zlabel('fz')

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(torques[:,0], torques[:,1], torques[:,2], c='b', s=50)
            ax.scatter(0, 0, 0, c='k', s=80)
            ax.set_xlim3d(-1.5, 1.5)
            ax.set_ylim3d(-1.5, 1.5)
            ax.set_zlim3d(-1.5, 1.5)
            ax.set_xlabel('tx')
            ax.set_ylabel('ty')
            ax.set_zlabel('tz')
            plt.show()

        if len(hull.vertices) == 0:
            logging.warning('Convex hull could not be computed')
            return 0.0

        # determine whether or not zero is in the convex hull
        s = time.time()
        min_norm_in_hull, v = PointGraspMetrics3D.min_norm_vector_in_facet(G, wrench_regularizer=wrench_regularizer)
        e = time.time()
        logging.debug('Min norm took %.3f sec' %(e - s))

        # if norm is greater than 0 then forces are outside of hull
        if min_norm_in_hull > wrench_norm_thresh:
            logging.debug('Zero not in convex hull')
            return 0.0

        # if there are fewer nonzeros than D-1 (dim of space minus one)
        # then zero is on the boundary and therefore we do not have
        # force closure
        if np.sum(v > 1e-4) <= G.shape[0]-1:
            logging.debug('Zero not in interior of convex hull')
            return 0.0

        # find minimum norm vector across all facets of convex hull
        s = time.time()
        min_dist = sys.float_info.max
        closest_facet = None
        facets = []
        for v in hull.vertices:
            if np.max(np.array(v)) < G.shape[1]: # because of some occasional odd behavior from pyhull
                facet = G[:, v]
                facets.append(facet)
                dist, _ = PointGraspMetrics3D.min_norm_vector_in_facet(facet, wrench_regularizer=wrench_regularizer)
                if dist < min_dist:
                    min_dist = dist
                    closest_facet = v
        e = time.time()
        logging.debug('Min dist took %.3f sec for %d vertices' %(e - s, len(hull.vertices)))

        return min_dist

    @staticmethod
    def wrench_in_positive_span(wrench_basis, target_wrench, force_limit, num_fingers=1,
                                wrench_norm_thresh = 1e-4, wrench_regularizer = 1e-10):
        """ Check whether a target can be exerted by positive combinations of wrenches in a given basis with L1 norm fonger force limit limit.

        Parameters
        ----------
        wrench_basis : 6xN :obj:`numpy.ndarray`
            basis for the wrench space
        target_wrench : 6x1 :obj:`numpy.ndarray`
            target wrench to resist
        force_limit : float
            L1 upper bound on the forces per finger (aka contact point)
        num_fingers : int
            number of contacts, used to enforce L1 finger constraint
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        int
            whether or not wrench can be resisted
        float
            minimum norm of the finger forces required to resist the wrench
        """
        num_wrenches = wrench_basis.shape[1]

        # quadratic and linear costs
        P = wrench_basis.T.dot(wrench_basis) + wrench_regularizer*np.eye(num_wrenches)
        q = -wrench_basis.T.dot(target_wrench)

        # inequalities
        lam_geq_zero = -1 * np.eye(num_wrenches)

        num_wrenches_per_finger = num_wrenches / num_fingers
        force_constraint = np.zeros([num_fingers, num_wrenches])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            force_constraint[i, start_i:end_i] = np.ones(num_wrenches_per_finger)

        G = np.r_[lam_geq_zero, force_constraint]
        h = np.zeros(num_wrenches+num_fingers)
        for i in range(num_fingers):
            h[num_wrenches+i] = force_limit

        # convert to cvx and solve
        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        sol = cvx.solvers.qp(P, q, G, h)
        v = np.array(sol['x'])

        min_dist = np.linalg.norm(wrench_basis.dot(v).ravel() - target_wrench)**2

        # add back in the target wrench
        return min_dist < wrench_norm_thresh, v

    @staticmethod
    def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
        """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

        Parameters
        ----------
        facet : 6xN :obj:`numpy.ndarray`
            vectors forming the facet
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float
            minimum norm of any point in the convex hull of the facet
        Nx1 :obj:`numpy.ndarray`
            vector of coefficients that achieves the minimum
        """
        dim = facet.shape[1] # num vertices in facet

        # create alpha weights for vertices of facet
        G = facet.T.dot(facet)
        grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
        P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
        q = cvx.matrix(np.zeros((dim, 1)))
        G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(1))         # combinations of vertices

        sol = cvx.solvers.qp(P, q, G, h, A, b)
        v = np.array(sol['x'])
        min_norm = np.sqrt(sol['primal objective'])

        return abs(min_norm), v

    @staticmethod
    def min_norm_vector_among_facets(facets, wrench_regularizer=1e-10):
        """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

        Parameters
        ----------
        facet : 6xN :obj:`numpy.ndarray`
            vectors forming the facet
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float
            minimum norm of any point in the convex hull of the facet
        Nx1 :obj:`numpy.ndarray`
            vector of coefficients that achieves the minimum
        """
        # read params
        num_facets = len(facets)
        if num_facets == 0:
            raise ValueError('Need to supply at least one facet!')
        facet_dim = facets[0].shape[1]
        dim = num_facets * facet_dim

        # init G matrix
        G = np.zeros([dim, dim])
        cur_i = 0
        end_i = cur_i + facet_dim
        for facet in facets:
            G[cur_i:end_i,cur_i:end_i] = facet.T.dot(facet)
            cur_i = cur_i + facet_dim
            end_i = cur_i + facet_dim
        grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

        # init equality constraint
        M = np.zeros([num_facets, dim])
        cur_i = 0
        end_i = cur_i + facet_dim
        for k, facet in enumerate(facets):
            M[k,cur_i:end_i] = np.ones(facet_dim)
            cur_i = cur_i + facet_dim
            end_i = cur_i + facet_dim

        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
        P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
        q = cvx.matrix(np.zeros((dim, 1)))
        G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(M)  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(num_facets))         # combinations of vertices

        sol = cvx.solvers.qp(P, q, G, h, A, b)
        v = np.array(sol['x'])

        # eval min norm
        cur_i = 0
        end_i = cur_i + facet_dim
        min_dist = np.inf
        best_facet = -1
        for k, facet in enumerate(facets):
            alpha = v[cur_i:end_i]
            dist = np.sqrt(alpha.T.dot(facet.T.dot(facet)).dot(alpha))
            if np.abs(np.sum(alpha) - 1) > 1e-3:
                continue
            if dist < min_dist:
                min_dist = dist
                best_facet = k
        return min_dist, v