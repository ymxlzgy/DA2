"""
Class to encapsulate robot grippers
Author: Jeff
"""
import json
import numpy as np
import os
import sys

import IPython

import meshpy.obj_file as obj_file

from autolab_core import RigidTransform

GRIPPER_MESH_FILENAME = 'gripper.obj'
FINGER_MESH_FILENAME = 'finger.obj'
HAND_MESH_FILENAME = 'hand.obj'
GRIPPER_PARAMS_FILENAME = 'params.json'
T_MESH_GRIPPER_FILENAME = 'T_mesh_gripper.tf'
T_GRASP_GRIPPER_FILENAME = 'T_grasp_gripper.tf'

class RobotGripper(object):
    """ Robot gripper wrapper for collision checking and encapsulation of grasp parameters (e.g. width, finger radius, etc)
    Note: The gripper frame should be the frame used to command the physical robot

    Attributes
    ----------
    name : :obj:`str`
        name of gripper
    mesh : :obj:`Mesh3D`
        3D triangular mesh specifying the geometry of the gripper
    params : :obj:`dict`
        set of parameters for the gripper, at minimum (finger_radius and grasp_width)
    T_mesh_gripper : :obj:`RigidTransform`
        transform from mesh frame to gripper frame (for rendering)
    T_grasp_gripper : :obj:`RigidTransform`
        transform from gripper frame to the grasp canonical frame (y-axis = grasp axis, x-axis = palm axis)
    """

    def __init__(self, name, mesh, mesh_filename, params, T_mesh_gripper, T_grasp_gripper):
        self.name = name
        self.mesh = mesh
        self.type = "parallel_jaw"
        self.mesh_filename = mesh_filename
        self.T_mesh_gripper = T_mesh_gripper
        self.T_grasp_gripper = T_grasp_gripper
        for key, value in params.items():
            propobj = getattr(self.__class__, key, None)
            if isinstance(propobj, property):
                if propobj.fset is None:
                    raise AttributeError("Can't set attribute {} from params in RobotGripper".format(key))
                propobj.fset(self, value)
            else:
                setattr(self, key, value)

    def collides_with_table(self, grasp, stable_pose, clearance=0.0):
        """ Checks whether or not the gripper collides with the table in the stable pose.
        No longer necessary with CollisionChecker.

        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp parameterizing the pose of the gripper
        stable_pose : :obj:`StablePose`
            specifies the pose of the table
        clearance : float
            min distance from the table

        Returns
        -------
        bool
            True if collision, False otherwise
        """
        # transform mesh into object pose to check collisions with table
        T_obj_gripper = grasp.gripper_pose(self)

        T_obj_mesh = T_obj_gripper * self.T_mesh_gripper.inverse()
        mesh_tf = self.mesh.transform(T_obj_mesh.inverse())

        # extract table
        n = stable_pose.r[2,:]
        x0 = stable_pose.x0

        # check all vertices for intersection with table
        collision = False
        for vertex in mesh_tf.vertices():
            v = np.array(vertex)
            if n.dot(v - x0) < clearance:
                collision = True
        return collision

    @staticmethod
    def load(gripper_name, gripper_dir='data/dex_grippers'):
        """ Load the gripper specified by gripper_name.

        Parameters
        ----------
        gripper_name : :obj:`str`
            name of the gripper to load
        gripper_dir : :obj:`str`
            directory where the gripper files are stored

        Returns
        -------
        :obj:`RobotGripper`
            loaded gripper objects
        """
        mesh_filename = os.path.join(gripper_dir, gripper_name, GRIPPER_MESH_FILENAME)
        mesh = obj_file.ObjFile(mesh_filename).read()

        f = open(os.path.join(os.path.join(gripper_dir, gripper_name, GRIPPER_PARAMS_FILENAME)), 'r')
        params = json.load(f)

        T_mesh_gripper = RigidTransform.load(os.path.join(gripper_dir, gripper_name, T_MESH_GRIPPER_FILENAME))
        T_grasp_gripper = RigidTransform.load(os.path.join(gripper_dir, gripper_name, T_GRASP_GRIPPER_FILENAME))
        return RobotGripper(gripper_name, mesh, mesh_filename, params, T_mesh_gripper, T_grasp_gripper)

class ParametrizedParallelJawGripper(RobotGripper):
    """ A robot gripper that has a parameterized geometry. """
    def __init__(self, name, finger_mesh, hand_mesh, mesh_filename, params, fingertip_x=0.1, fingertip_y=0.1, palm_depth=0.2, max_width=0.1, gripper_offset=0.01):
        # init params
        self.mesh = None
        self._finger_mesh = finger_mesh
        self._hand_mesh = hand_mesh
        self._fingertip_x = fingertip_x
        self._fingertip_y = fingertip_y
        self._palm_depth = palm_depth
        self._max_width = max_width
        self._gripper_offset = gripper_offset

        R_grasp_gripper = np.array([[0, 0, 1],
                                    [1, 0, 0],
                                    [0, 1, 0]])
        self.T_grasp_gripper = RigidTransform(rotation=R_grasp_gripper,
                                              from_frame='gripper', to_frame='grasp')
        self.T_mesh_gripper = RigidTransform(from_frame='gripper', to_frame='mesh')
        self.mesh_filename = mesh_filename
        self._generate_mesh()
        RobotGripper.__init__(self, name, self.mesh, mesh_filename, params, self.T_mesh_gripper, self.T_grasp_gripper)

    @property
    def fingertip_x(self):
        return self._fingertip_x

    @property
    def fingertip_y(self):
        return self._fingertip_y

    @property
    def palm_depth(self):
        return self._palm_depth

    @property
    def width(self):
        return self._max_width

    @property
    def max_width(self): # To maintain compatiblity with RobotGripper
        return self.width

    @property
    def gripper_offset(self):
        return self._gripper_offset

    @fingertip_x.setter
    def fingertip_x(self, val):
        self._fingertip_x = val
        self._generate_mesh()

    @fingertip_y.setter
    def fingertip_y(self, val):
        self._fingertip_y = val
        self._generate_mesh()

    @palm_depth.setter
    def palm_depth(self, val):
        self._palm_depth = val
        self._generate_mesh()

    @width.setter
    def width(self, val):
        self._max_width = val
        self._generate_mesh()

    @max_width.setter
    def max_width(self, val): # To maintain compatiblity with RobotGripper
        self.width = val

    def update(self, fingertip_x, fingertip_y, palm_depth, width, gripper_offset=None):
        self._fingertip_x = fingertip_x
        self._fingertip_y = fingertip_y
        self._palm_depth = palm_depth
        self._max_width = width
        if gripper_offset:
            self._gripper_offset = gripper_offset
        self._generate_mesh()

    def _generate_mesh(self):
        """ Load the geometry for the fingers and palm separately so that they can be rescaled later. """
        # rescale palm and fingers
        finger = self._finger_mesh.copy()
        finger.vertices[:,0] = self._fingertip_x * finger.vertices[:,0]
        finger.vertices[:,1] = self._fingertip_y * finger.vertices[:,1]
        finger.vertices[:,2] = self._palm_depth * finger.vertices[:,2]

        hand = self._hand_mesh.copy()
        hand_width = 4 * self._fingertip_x + self._max_width
        hand.vertices[:,0] = hand_width * hand.vertices[:,0]
        hand.vertices[:,1] = self._fingertip_y * hand.vertices[:,1]
        hand.vertices[:,2] = self._palm_depth * hand.vertices[:,2]

        # transform meshes
        t_finger1_gripper = [self._max_width / 2.0 + self._fingertip_x / 2.0,
                             0,
                             self._palm_depth / 2.0]
        t_finger2_gripper = [-self._max_width / 2.0 - self._fingertip_x / 2.0,
                             0,
                             self._palm_depth / 2.0]
        t_hand_gripper    = [0,
                             0,
                             -self._palm_depth / 2.0]
        T_finger1_gripper = RigidTransform(translation=t_finger1_gripper,
                                           rotation=np.array([[-1,  0,  0],
                                                              [ 0, -1,  0],
                                                              [ 0,  0,  1]]),
                                           from_frame='finger1',
                                           to_frame='gripper')
        T_finger2_gripper = RigidTransform(translation=t_finger2_gripper,
                                           from_frame='finger2',
                                           to_frame='gripper')
        T_hand_gripper = RigidTransform(translation=t_hand_gripper,
                                        from_frame='hand',
                                        to_frame='gripper')
        finger1 = finger.transform(T_finger1_gripper)
        finger2 = finger.transform(T_finger2_gripper)
        hand_tf = hand.transform(T_hand_gripper)

        # zip meshes together
        fingers = finger1.merge(finger2)
        self.mesh = hand_tf.merge(fingers)
        obj_file.ObjFile(self.mesh_filename).write(self.mesh)

        # update transforms
        self.T_mesh_gripper.translation[2] = self._palm_depth - self._gripper_offset

    @staticmethod
    def load(gripper_name='generic', gripper_dir='data/grippers'):
        """ Load the gripper specified by gripper_name.

        Parameters
        ----------
        gripper_name : :obj:`str`
            name of the gripper to load
        gripper_dir : :obj:`str`
            directory where the gripper files are stored

        Returns
        -------
        :obj:`RobotGripper`
            loaded gripper objects
        """
        finger_mesh_filename = os.path.join(gripper_dir, gripper_name, FINGER_MESH_FILENAME)
        finger_mesh = obj_file.ObjFile(finger_mesh_filename).read()

        hand_mesh_filename = os.path.join(gripper_dir, gripper_name, HAND_MESH_FILENAME)
        hand_mesh = obj_file.ObjFile(hand_mesh_filename).read()

        f = open(os.path.join(os.path.join(gripper_dir, gripper_name, GRIPPER_PARAMS_FILENAME)), 'r')
        params = json.load(f)

        mesh_filename = os.path.join(gripper_dir, gripper_name, GRIPPER_MESH_FILENAME)

        return ParametrizedParallelJawGripper(gripper_name, finger_mesh, hand_mesh, mesh_filename, params)