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

        normals = np.array([[[0,0,1],[0,0,1]]])

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
        self.assertEqual(outputs['deflected_normals'].tolist(), [[[1,0,0],[0,0,1]]])

    def test_dual_surfaces_daisy_chain(self):
        inputs = {'delta_aileron': 90,
                  'def_mesh': self.mesh,
                  'undeflected_normals': self.normals}
        outputs = {'deflected_normals': None}

        control_surface_left = ControlSurface(mesh=self.mesh, cLoc=[0,0], yLoc=[0,1])
        control_surface_right = ControlSurface(mesh=self.mesh, cLoc=[0,0], yLoc=[1,2])
        control_surface_left.setup()
        control_surface_right.setup()

        # deflect left
        control_surface_left.compute(inputs, outputs)
        self.assertEqual(outputs['deflected_normals'].tolist(), [[[1,0,0],[0,0,1]]])

        # deflect right
        inputs['undeflected_normals'] = outputs['deflected_normals']  # daisy chain
        control_surface_right.compute(inputs, outputs)
        self.assertEqual(outputs['deflected_normals'].tolist(), [[[1,0,0],[1,0,0]]])


class TestFourPanelMesh(unittest.TestCase):
    def setUp(self):
        # Four panel mesh on the xy plane
        #
        # (0,0)------(0,1)------(0,2)------(0,3)------(0,4)
        #   |          |          |          |          |  
        #   |          |          |          |          |  
        # (1,0)------(1,1)------(1,2)------(1,3)------(1,4)
        #   
        #  |-->y
        # x|
        #  v
        #

        mesh = np.array([[(0,0,0), (0,1,0), (0,2,0), (0,3,0), (0,4,0)],
                         [(1,0,0), (1,1,0), (1,2,0), (1,3,0), (1,4,0)]])

        normals = np.tile(np.array([0,0,1]), (1,4,1))

        self.mesh = mesh
        self.normals = normals

    def test_inboard_antisymmetric(self):
        inputs = {'delta_aileron': 90,
                  'def_mesh': self.mesh,
                  'undeflected_normals': self.normals}
        outputs = {'deflected_normals': None}

        control_surface = ControlSurface(mesh=self.mesh, cLoc=[0,0], yLoc=[1,2], antisymmetric=True)
        control_surface.setup()
        control_surface.compute(inputs, outputs)

        self.assertEqual(outputs['deflected_normals'].tolist(), [[[0,0,1],[1,0,0],[-1,0,0],[0,0,1]]])

    def test_outboard_antisymmetric(self):
        inputs = {'delta_aileron': 90,
                  'def_mesh': self.mesh,
                  'undeflected_normals': self.normals}
        outputs = {'deflected_normals': None}

        control_surface = ControlSurface(mesh=self.mesh, cLoc=[0,0], yLoc=[0,1], antisymmetric=True)
        control_surface.setup()
        control_surface.compute(inputs, outputs)

        self.assertEqual(outputs['deflected_normals'].tolist(), [[[1,0,0],[0,0,1],[0,0,1],[-1,0,0]]])


if __name__=='__main__':
    unittest.main()
