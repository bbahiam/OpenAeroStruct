from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder,\
                             ExplicitComponent, n2
from openaerostruct.utils.constants import grav_constant

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 11,
             'num_x' : 5,
             'wing_type' : 'rect',
             'symmetry' : False,
             'span' : 10.,
             'chord' : 1.}

mesh = generate_mesh(mesh_dict)


aileron = {
       'name': 'aileron',       # Name of control surface
       'yLoc': [0,2],           # Spanwise mesh indices of start/end locations
       'cLoc': [0.75,0.75],     # Chordwise position of the hingeline
       'antisymmetric': True,   # Antisymmetry
       'corrector' : True       # Semi-empirical corrector for lift effectiveness
       }                        # (best when aileron is a small % the spanwise length)




surface = {
            'control_surfaces' : [aileron], # control surface list
            
            
            
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : False,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'thickness_cp' : np.array([.3, .3, .3]),

            'twist_cp' : np.array([0.]),
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
            't_over_c_cp' : np.array([0.15]),      # thickness over chord ratio (NACA0015)
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
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
            }

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()

#
#
#
# Aileron deflection angle
indep_var_comp.add_output('delta_aileron', val=12.5,units='deg')
#
#
#

indep_var_comp.add_output('cg', val=np.array([0.,0.,0.]),units='m')
indep_var_comp.add_output('v', val=248.136, units='m/s')
indep_var_comp.add_output('alpha', val=5., units='deg')
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
indep_var_comp.add_output('omega', val=np.array([0.,0.,0.]),units='rad/s')

prob.model.add_subsystem('prob_vars',
                 indep_var_comp,
                 promotes=['*'])
                
aerostruct_group = AerostructGeometry(surface=surface)
        
name = 'wing'
                
# Add tmp_group to the problem with the name of the surface.
prob.model.add_subsystem(name, aerostruct_group)
                
point_name = 'AS_point_0'
                
# Create the aero point group and add it to the model
# Rotational=true, rollOnly=True skips over Breguet range, weight calc, cg calc, etc.
AS_point = AerostructPoint(surfaces=[surface],rotational=True,rollOnly=True)
        
prob.model.add_subsystem(point_name, AS_point,
            promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg', 'delta_aileron','omega'])
                
com_name = point_name + '.' + name + '_perf'
    
prob.model.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')
                
# Connect aerodyamic mesh to coupled group mesh
prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')
                
# Connect performance calculation variables
prob.model.connect(name + '.radius', com_name + '.radius')
prob.model.connect(name + '.thickness', com_name + '.thickness')
prob.model.connect(name + '.nodes', com_name + '.nodes')
prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')
        

## Moment balance constraint
class momentBalance(ExplicitComponent):
    def initialize(self):
        self.options.declare('surface', types=list)
        
    def setup(self):
        self.add_input('CM', val=np.array([1.,1.,1.]))
        self.add_output('residual', val=1.0)
        
        self.declare_partials('residual','CM')
        
    def compute(self,inputs,outputs):
        M_x = inputs['CM']
        outputs['residual'] = np.abs(M_x[0])
    
    def compute_partials(self,inputs,partials):
        M_x = inputs['CM']
        partials['residual','CM'] = np.array([M_x[0]/np.abs(M_x[0]),0.,0.])
        
##############################################################################
#                    SETUP OPTIMIZER to DRIVE CM_x -> 0                      #
##############################################################################
        
myMomBal = momentBalance(surface=[surface])
#prob.model.add_subsystem('balanceEqn', myMomBal,promotes=['*'])#,promotes_inputs=['CM'],promotes_outputs['residual'])
    
prob.model.add_subsystem('balanceEqn', myMomBal,promotes_inputs=['CM'],
                         promotes_outputs=['residual'])#,promotes_inputs=['CM'],promotes_outputs['residual'])
        
prob.model.connect(point_name+'.CM', 'CM')
    
from openmdao.api import ScipyOptimizeDriver
prob.driver = ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-9
        
prob.model.add_design_var('omega', lower=np.array([-5.,0.,0.]), upper=np.array([5.,0.,0.]))
prob.model.add_objective('residual')
        
# Set up the problem
prob.setup(check=True)

# Only run analysis
# prob.run_model()

# Run optimization
prob.run_driver()

print()
print('CL:', prob['AS_point_0.wing_perf.CL'])
print('CD:', prob['AS_point_0.wing_perf.CD'])
print('CM:', prob['AS_point_0.CM'])