from __future__ import division, print_function

from openmdao.api import Group

from openaerostruct.functionals.breguet_range import BreguetRange
from openaerostruct.functionals.equilibrium import Equilibrium
from openaerostruct.functionals.center_of_gravity import CenterOfGravity
from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.functionals.sum_areas import SumAreas

class TotalRollPerformance(Group):
    """
    Group to contain the total aerostructural performance components.
    """
    # TODO add more than just moment, cl/cd
    # expand this area to other roll-specific functions
    def initialize(self):
        self.options.declare('surfaces', types=list)
        self.options.declare('user_specified_Sref', types=bool)
        self.options.declare('internally_connect_fuelburn', types=bool, default=True)
        self.options.declare('rotational',types=bool,default=False)
        
    def setup(self):
        surfaces = self.options['surfaces']
        
        if not self.options['user_specified_Sref']:
            self.add_subsystem('sum_areas',
                SumAreas(surfaces=surfaces),
                promotes_inputs=['*S_ref'],
                promotes_outputs=['S_ref_total'])

        self.add_subsystem('CL_CD',
             TotalLiftDrag(surfaces=surfaces),
             promotes_inputs=['*CL', '*CD', '*S_ref', 'S_ref_total'],
             promotes_outputs=['CL', 'CD'])

        self.add_subsystem('moment',
             MomentCoefficient(surfaces=surfaces),
             promotes_inputs=['v', 'rho', 'cg', 'S_ref_total', '*b_pts', '*widths', '*chords', '*sec_forces', '*S_ref'],
             promotes_outputs=['CM'])