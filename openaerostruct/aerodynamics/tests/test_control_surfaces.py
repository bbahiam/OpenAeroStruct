import unittest

import numpy as np

from openaerostruct.aerodynamics.control_surfaces import ControlSurface, find_hinge


class TestSinglePanelMesh(unittest.TestCase):
    def setUp(self):
        # One panel mesh on the xy plane
        #
        # (0,0)------(0,1)
        #   |          |
        #   |          |
        # (1,0)------(1,1)
        #   
        #  |-->y
        # x|
        #  v
        #
        mesh = np.zeros((2,2,3))
        mesh[0,0,:] = 0, 0, 0
        mesh[0,1,:] = 0, 1, 0
        mesh[1,1,:] = 1, 1, 0
        mesh[1,0,:] = 1, 0, 0

        normal = np.array((0,0,1))
        normals = normal[np.newaxis, np.newaxis, :]

        self.mesh = mesh
        self.normals = normals

    def test_find_hinge(self):
        h0, h1, hinge = find_hinge(mesh=self.mesh, yLoc=[0,1], cLoc=[0,0])
        self.assertEqual(h0.tolist(), [0,0,0])
        self.assertEqual(h1.tolist(), [0,1,0])
        self.assertEqual(hinge.tolist(), [0,1,0])

    def test_90deg_deflection(self):
        inputs = {'delta_aileron': 90,
                  'def_mesh': self.mesh,
                  'undeflected_normals': self.normals}
        outputs = {'deflected_normals': None}

        control_surface = ControlSurface(mesh=self.mesh, cLoc=[0,0], yLoc=[0,1])
        control_surface.setup()
        control_surface.compute(inputs, outputs)
        self.assertEqual(outputs['deflected_normals'].tolist(), [[[1,0,0]]])


class TestDualPanelMesh(unittest.TestCase):
    def setUp(self):
        # Two panel mesh on the xy plane
        #
        # (0,0)------(0,1)------(0,2)
        #   |          |          |  
        #   |          |          |  
        # (1,0)------(1,1)------(1,2)
        #   
        #  |-->y
        # x|
        #  v
        #
        mesh = np.zeros((2,3,3))
        mesh[0,0,:] = 0, 0, 0
        mesh[0,1,:] = 0, 1, 0
        mesh[0,2,:] = 0, 2, 0
        mesh[1,2,:] = 1, 2, 0
        mesh[1,1,:] = 1, 1, 0
        mesh[1,0,:] = 1, 0, 0

        normals = np.array([[[0,0,1]],[[0,0,1]]])

        self.mesh = mesh
        self.normals = normals

    def test_find_hinge(self):
        h0, h1, hinge = find_hinge(mesh=self.mesh, yLoc=[0,1], cLoc=[0,0])
        self.assertEqual(h0.tolist(), [0,0,0])
        self.assertEqual(h1.tolist(), [0,1,0])
        self.assertEqual(hinge.tolist(), [0,1,0])

    def test_90deg_deflection(self):
        inputs = {'delta_aileron': 90,
                  'def_mesh': self.mesh,
                  'undeflected_normals': self.normals}
        outputs = {'deflected_normals': None}

        # Control surface on left panel
        control_surface = ControlSurface(mesh=self.mesh, cLoc=[0,0], yLoc=[0,1])
        control_surface.setup()
        control_surface.compute(inputs, outputs)
        self.assertEqual(outputs['deflected_normals'].tolist(), [[[1,0,0]],[[0,0,1]]])


if __name__=='__main__':
    unittest.main()
