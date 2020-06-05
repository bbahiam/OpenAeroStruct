import itertools

import openmdao.api as om

from openaerostruct.aerodynamics.control_surfaces import ControlSurface


def _pairwise(iterable):  # From itertools recipes
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class ControlSurfacesGroup(om.Group):
    """ A collection of control surfaces generated from a list of dictionaries"""

    def initialize(self):
        self.options.declare('control_surfaces', types=list)
        self.options.declare('mesh')

    def setup(self):
        control_surfaces = self.options['control_surfaces']

        # Add control surfaces as subsystems
        for control_surface in control_surfaces:
            control_surface_component = ControlSurface(
                mesh=self.options['mesh'],
                yLoc=control_surface['yLoc'],
                cLoc=control_surface['cLoc'],
                antisymmetric=control_surface['antisymmetric'],
                semi_empirical_correction=control_surface['corrector'],
                )

            self.add_subsystem(control_surface['name'], control_surface_component)

        # Connect control surfaces
        self.promotes(control_surfaces[0]['name'], inputs=['undeflected_normals'])
        for surf1, surf2 in _pairwise(control_surfaces):
            self.connect(surf1['name']+'.deflected_normals',
                         surf2['name']+'.undeflected_normals')
        self.promotes(control_surfaces[-1]['name'], outputs=['deflected_normals'])

        # Promote def_mesh for all surfaces
        for surf in control_surfaces:
            self.promotes(surf['name'], inputs=['def_mesh'])


if __name__ =='__main__':
    import numpy as np
    from openaerostruct.geometry.utils import generate_mesh

    # Based on NACA TN2563
    # Brief description
    S = 2*0.092903
    AR = 8
    b = np.sqrt(S*AR)
    taper = 0.45
    rc = 2*S/(b*(taper+1))
    tc = rc*taper
    sweep = 46 # from LE
    cg_loc = np.array([0.38*rc,0,0])

    # Testing conditions
    alpha = 0.012 # From Kirsten Wind Tunnel page
    rho = 1.2

    n = 4
    num_y = n*(10)+1
    num_x = 7

    # Create a dictionary to store options about the surface
    mesh_dict = {'num_y' : num_y,
                 'num_x' : num_x,
                 'wing_type' : 'rect',
                 'symmetry' : False,
                 'span' : b,
                 'root_chord' : rc}

    mesh = generate_mesh(mesh_dict)
    control_surfaces = [
        {
            'name': 'ail0'+str(ind[1]),
            'yLoc': list(ind),
            'cLoc': [0.7, 0.7], #All ailerons are 0.3c
            'antisymmetric': True,
            'corrector': True
        } for ind in _pairwise([0, 2, 4, 6, 8])]

    csg = ControlSurfacesGroup(control_surfaces=control_surfaces, mesh=mesh)

    p = om.Problem()
    p.model.add_subsystem('control_surfaces', csg)
    p.setup()
    p.final_setup()
