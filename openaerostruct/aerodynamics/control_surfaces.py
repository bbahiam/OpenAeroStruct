from __future__ import print_function, division
import numpy as np
from scipy.spatial.transform import Rotation as R

from openmdao.api import ExplicitComponent


class ControlSurface(ExplicitComponent):

    def initialize(self):
        self.options.declare('surface', types=dict)  # Is this needed?
        self.options.declare('panels', types=list)

    def setup(self):
        self.surface = surface = self.options['surface']
        self.panels = panels = self.options['panels']

        mesh = surface['mesh']
        nx = self.nx = mesh.shape[0]
        ny = self.ny = mesh.shape[1]

        self.add_input('delta_aileron', val=0)
        self.add_input('normals', val=np.zeros((nx-1, ny-1, 3)))
        self.add_output('deflected_normals', val=np.zeros((nx-1, ny-1, 3)))

        rows, cols = panels
        self.declare_partials('deflected_normals',
                              'normals', 
                              #rows=rows, cols=cols, 
                              method='fd')

    def compute(self, inputs, outputs):
        deflection = inputs['delta_aileron'] #TODO:change name
        normals = inputs['normals']

        # Actual code
        # Filter control surface normals
        rotation = self._get_rotation(inputs)
        # deflect normals
        new_normals = normals
        new_normals[self.panels] = rotation.apply(normals[self.panels])

        outputs['deflected_normals'] = new_normals

    def compute_partials(self, inputs, partials):
        # zero everywhere but on the new normal, where the jacobian is the rotation matrix
        rotation = self._get_rotation(inputs)
        partials = rotation.as_matrix()

    def _get_rotation(self, inputs):
        # get hinge line
        # rotation matrix
        hinge_line=np.array([0,1,0])
        rotation = R.from_rotvec(hinge_line*inputs['delta_aileron'])

        return rotation
