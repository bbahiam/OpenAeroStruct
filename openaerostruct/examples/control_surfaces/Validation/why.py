from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder
from openaerostruct.utils.constants import grav_constant

import pdb
from plotting import meshPlot

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 13,
             'num_x' : 7,
             'wing_type' : 'rect',
             'symmetry' : False,
             'span' : 1.2192,
             'chord' : 7*0.0254}

mesh = generate_mesh(mesh_dict)

aileron = {
           'name': 'aileron',
           'yLoc': [1,2],
           'cLoc': [0.7,0.7],
           'antisymmetric' : True
           }

surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : False,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'thickness_cp' : np.array([.1]),
            'radius_cp' : np.array([.1]),
            'twist_cp' : np.array([0.]),
            'mesh' : mesh,
            'taper' : 0.45,
            'control_surfaces': [aileron],

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.0,            # CL of the surface at alpha=0
            'CD0' : 0.015,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c_cp' : np.array([0.12]),      # thickness over chord ratio (NACA0015)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness
            'with_viscous' : True,
            'with_wave' : False,     # if true, compute wave drag

            # Structural values are based on aluminum 7075
            'E' : 120.e9,            # [Pa] Young's modulus of the spar
            'G' : 2.2e9,            # [Pa] shear modulus of the spar
            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 3.e3,          # [kg/m^3] material density
            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            'wing_weight_ratio' : 1.25,
            'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
            'distributed_fuel_weight' : False,
            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            }

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=248.136, units='m/s')
indep_var_comp.add_output('alpha', val=5., units='deg')
indep_var_comp.add_output('Mach_number', val=0.84)
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
indep_var_comp.add_output('CT', val=grav_constant * 17.e-6, units='1/s')
indep_var_comp.add_output('R', val=0., units='m')
indep_var_comp.add_output('W0', val=0.4 * 3e5,  units='kg')
indep_var_comp.add_output('speed_of_sound', val=295.4, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')
indep_var_comp.add_output('delta_aileron', val=12.,units='deg')

prob.model.add_subsystem('prob_vars',
     indep_var_comp,
     promotes=['*'])

aerostruct_group = AerostructGeometry(surface=surface)

name = 'wing'

# Add tmp_group to the problem with the name of the surface.
prob.model.add_subsystem(name, aerostruct_group)

point_name = 'AS_point_0'

# Create the aero point group and add it to the model
AS_point = AerostructPoint(surfaces=[surface])

prob.model.add_subsystem(point_name, AS_point,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
        'W0', 'speed_of_sound', 'empty_cg', 'load_factor', 'delta_aileron'])

com_name = point_name + '.' + name + '_perf'
prob.model.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

# Connect aerodyamic mesh to coupled group mesh
prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

# Connect performance calculation variables
prob.model.connect(name + '.radius', com_name + '.radius')
prob.model.connect(name + '.thickness', com_name + '.thickness')
prob.model.connect(name + '.nodes', com_name + '.nodes')
prob.model.connect(name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location')
prob.model.connect(name + '.structural_mass', point_name + '.' + 'total_perf.' + name + '_structural_mass')
prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')

from openmdao.api import ScipyOptimizeDriver
prob.driver = ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-9

#recorder = SqliteRecorder("aerostruct.db")
#prob.driver.add_recorder(recorder)
#prob.driver.recording_options['record_derivatives'] = True
#prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables, constraint, and objective
prob.model.add_design_var('wing.twist_cp', lower=-10., upper=15.)
prob.model.add_design_var('wing.thickness_cp', lower=0.01, upper=0.5, scaler=1e2)
prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)
prob.model.add_constraint('AS_point_0.wing_perf.thickness_intersects', upper=0.)

# Add design variables, constraisnt, and objective on the problem
prob.model.add_design_var('alpha', lower=-10., upper=10.)
prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)
prob.model.add_objective('AS_point_0.fuelburn', scaler=1e-5)

# Set up the problem

dels = np.array([-10.,-5.,0.,5.,10.])
CMs = np.zeros_like(dels)

counter = 0
for i,da in enumerate(dels):
    prob.setup(check=True)
    prob.model.AS_point_0.coupled.nonlinear_solver.options['maxiter'] = 300
    
    prob['alpha'] = 0.012
    prob['v'] = 44.6655628
    prob['delta_aileron'] = da
    prob.run_model()
    
    meshPlot(prob['AS_point_0.coupled.wing.def_mesh'],deformed=True,\
             name=None,azim=90,elev=0)
    print(prob['AS_point_0.total_perf.moment.CM'][0])

#vels = np.array([28.24898228, 34.59779617, 39.95009387, 44.6655628 , 48.92867258,
#       52.84900661, 56.49796457, 59.9251408 , 63.16664468, 66.24973585,
#       69.19559235])
#dels = np.array([-10.,-5.,0.,5.,10.])
#
#D,V = np.meshgrid(dels,vels)
#
#CM = np.ones((len(vels),len(dels)))
#
#for i,v in enumerate(vels):
#    for j,d in enumerate(dels):
#        prob['v'] = v
#        prob['Mach_number'] = v/prob['speed_of_sound']
#        prob['delta_aileron'] = d
#        
#        prob['v'] = 100.
#        prob['Mach_number'] = v/prob['speed_of_sound']
#        prob['delta_aileron'] = 12.5
#        # Run the model
#        prob.run_model()
#        
#        CM[i,j] = prob['AS_point_0.total_perf.moment.CM'][0]     