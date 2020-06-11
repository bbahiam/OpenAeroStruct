import unittest

import numpy as np
import openmdao.api as om

from openaerostruct.aerodynamics.control_surfaces_group import ControlSurfacesGroup


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

        control_surfaces = [
            {'name': 'left',
             'cLoc': [0, 0],
             'yLoc': [0, 1]},
            {'name': 'right',
             'cLoc': [0, 0],
             'yLoc': [1, 2]},
        ]

        control_surfaces_group = ControlSurfacesGroup(mesh=mesh,
                                                      control_surfaces=control_surfaces)

        indep_var = om.IndepVarComp()
        indep_var.add_output('def_mesh', mesh)
        indep_var.add_output('normals', normals)
        indep_var.add_output('left_delta', units='deg')
        indep_var.add_output('right_delta', units='deg')

        p = om.Problem()
        p.model.add_subsystem('indep_var', indep_var)
        p.model.add_subsystem('control_surfaces', control_surfaces_group)

        p.model.connect('indep_var.def_mesh', 'control_surfaces.def_mesh')
        p.model.connect('indep_var.normals', 'control_surfaces.undeflected_normals')
        p.model.connect('indep_var.left_delta', 'control_surfaces.left.delta_aileron')
        p.model.connect('indep_var.right_delta', 'control_surfaces.right.delta_aileron')

        p.setup()

        self.p = p

    def test_left(self):
        self.p['indep_var.left_delta'] = 90
        self.p['indep_var.right_delta'] = 0

        self.p.run_model()

        np.testing.assert_almost_equal(self.p['control_surfaces.deflected_normals'],
                                       np.array([[[1, 0, 0], [0, 0, 1]]]))

    def test_right(self):
        self.p['indep_var.left_delta'] = 0
        self.p['indep_var.right_delta'] = 90

        self.p.run_model()

        np.testing.assert_almost_equal(self.p['control_surfaces.deflected_normals'],
                                       np.array([[[0, 0, 1], [1, 0, 0]]]))

    def test_both(self):
        self.p['indep_var.left_delta'] = 90
        self.p['indep_var.right_delta'] = 90

        self.p.run_model()

        np.testing.assert_almost_equal(self.p['control_surfaces.deflected_normals'],
                                       np.array([[[1, 0, 0], [1, 0, 0]]]))


if __name__=='__main__':
    unittest.main()
