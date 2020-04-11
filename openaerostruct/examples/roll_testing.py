import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 31,
             'num_x' : 7,
             'symmetry' : False,
             'wing_type' : 'rect',
             'span' : 10.,
             'chord' : 1.}

mesh = generate_mesh(mesh_dict)

# Create a dictionary for the control surface
aileron = {
       'name': 'aileron',
       'yLoc': [0,2],
       'cLoc': [0.75,0.75]
       }
    
surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : False,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'thickness_cp' : np.array([.01]),

            'twist_cp' : np.zeros((1)),
            'taper' : 1.,
            'mesh' : mesh,

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
            't_over_c_cp' : np.array([0.12]),      # thickness over chord ratio (NACA0012)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0012)
                                    # thickness
            'with_viscous' : True,
            'with_wave' : False,     # if true, compute wave drag

            # Structural values are based on aluminum 7075
            'E' : 70.e9,            # [Pa] Young's modulus of the spar
            'G' : 30.e9,            # [Pa] shear modulus of the spar
            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 3.e3,          # [kg/m^3] material density
            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            'wing_weight_ratio' : 2.,
            'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
            'distributed_fuel_weight' : False,
            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            'control_surfaces' : [aileron]
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
indep_var_comp.add_output('R', val=11.165e6, units='m')
indep_var_comp.add_output('W0', val=0.4 * 3e5,  units='kg')
indep_var_comp.add_output('speed_of_sound', val=295.4, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')
indep_var_comp.add_output('delta_aileron', val=0.)
indep_var_comp.add_output('omega', val=np.array([200.,0.,0.]),units='deg/s')

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
        'W0', 'speed_of_sound', 'empty_cg', 'load_factor','delta_aileron'])

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

recorder = SqliteRecorder("aerostruct.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']

prob.setup(check=True)

dels = np.linspace(0.,-30.,2)
CM_x = []
CM_y = []
CM_z = []

for d in dels:
    prob['delta_aileron'] = d
    prob.run_model()
    CM_x.append(prob['AS_point_0.total_perf.moment.CM'][0])
    CM_y.append(prob['AS_point_0.total_perf.moment.CM'][1])
    CM_z.append(prob['AS_point_0.total_perf.moment.CM'][2])

import matplotlib.pyplot as plt
plt.plot(dels,CM_x)
plt.xlabel('Aileron deflection (deg)')
plt.ylabel('Roll Moment Coefficient')