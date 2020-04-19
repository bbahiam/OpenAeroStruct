from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder, \
                            ExecComp, NonlinearBlockGS, ExplicitComponent, n2
                            
from openaerostruct.utils.constants import grav_constant

"""
    What this example does:
    -Takes a control surface deflection angle and gives the corresponding
     steady-state roll rate.
    -Find roll rate for multiple deflections at multiple velocities
    -Calculate the control effectiveness at each velocity
    
    Overview:
        -Define mesh and control surface
        -Set up aerostructural point group (rotational=True)
        -Define class to compute the abs val of the moment about x-axis
        -Set roll rate to design variable and moment to objective
        -Converges when roll moment from aerodynamic damping due to roll speed
         cancels out moment caused by control surface deflection

    To get control effectiveness, run the optimizer for multiple deflections,
    flight conditions, etc.
    
"""

##############################################################################
#                              GEOMETRY SETUP                                #
##############################################################################
# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 11,
             'num_x' : 5,
             'wing_type' : 'rect',
             'symmetry' : False,
             'span' : 10,
             'chord' : 1}

mesh = generate_mesh(mesh_dict)

# Create a dictionary for the control surface
aileron = {
       'name': 'aileron',
       'yLoc': [1,2],       # Start and end chordwise mesh points for the surface
       'cLoc': [0.7,0.85],  # Local chord percentage of the hinge location
       'antisymmetric': True # Creates CS mirrored across XZ with opposite/equal deflections
       }

surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : False,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'thickness_cp' : np.array([0.1]),

            'twist_cp' : np.zeros((1)),
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
            
            # Control surface
            'control_surfaces' : [aileron]
            }



##############################################################################
#                           PROBLEM PARAMETERS                               #
##############################################################################
# Parameters
v = 248.136
alpha = 1.
beta = 0.
Mach_number = 0.01
re = 1.e6
rho = 1.2
CT = grav_constant * 17.e-6
R = 0.
W0 = 0.4 * 3e5
speed_of_sound = 344.0
load_factor = 1.
cg_val = np.array([0.5,0,0]) # CG position is moment reference position
delta_aileron = 5.

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=v, units='m/s')
indep_var_comp.add_output('alpha', val=alpha, units='deg')
indep_var_comp.add_output('beta', val=beta, units='deg')
indep_var_comp.add_output('Mach_number', val=Mach_number)
indep_var_comp.add_output('re', val=re, units='1/m')
indep_var_comp.add_output('rho', val=rho, units='kg/m**3')
indep_var_comp.add_output('CT', val=CT, units='1/s')
indep_var_comp.add_output('R', val=R, units='m')
indep_var_comp.add_output('W0', val=W0,  units='kg')
indep_var_comp.add_output('speed_of_sound', val=speed_of_sound, units='m/s')
indep_var_comp.add_output('load_factor', val=load_factor)
indep_var_comp.add_output('cg', val=cg_val, units='m')
indep_var_comp.add_output('delta_aileron', val=delta_aileron, units='deg')
indep_var_comp.add_output('omega', val=np.array([2.,0.,0.]), units='rad/s')

prob.model.add_subsystem('prob_vars',
     indep_var_comp,
     promotes=['*'])

aerostruct_group = AerostructGeometry(surface=surface)

name = 'wing'

# Add tmp_group to the problem with the name of the surface.
prob.model.add_subsystem(name, aerostruct_group)

point_name = 'AS_point_0'

# Create the aero point group and add it to the model
AS_point = AerostructPoint(surfaces=[surface],compressible=False,rotational=True)

prob.model.add_subsystem(point_name, AS_point,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
        'W0', 'speed_of_sound', 'cg', 'load_factor','delta_aileron','omega'])

com_name = point_name + '.' + name + '_perf'
prob.model.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

# Connect aerodyamic mesh to coupled group mesh
prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

# Connect performance calculation variables
prob.model.connect(name + '.radius', com_name + '.radius')
prob.model.connect(name + '.thickness', com_name + '.thickness')
prob.model.connect(name + '.nodes', com_name + '.nodes')
#prob.model.connect(name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location')
prob.model.connect(name + '.structural_mass', point_name + '.' + 'total_perf.' + name + '_structural_mass')
prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')



##############################################################################
#                    SETUP OPTIMIZER to DRIVE CM_x -> 0                      #
##############################################################################
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

prob.model.AS_point_0.coupled.nonlinear_solver.options['maxiter'] = 250

prob['delta_aileron'] = 10.
prob['omega'] = np.array([0.4,0.,0.])
prob['v'] = 75.

# View model
#n2(prob)

##############################################################################
#                          RUN and COLLECT DATA                              #
##############################################################################
# Set up arrays to get data
dels = np.linspace(0.,17.5,5)   # deflections, deg
vels = np.linspace(20,300,5)    # velocities, m/s
Mach = np.ones_like(vels)       # Mach numbers
q = 0.5*rho*vels**2             # Dynamic pressure

p = np.zeros((len(dels),len(vels))) # Angular velocity, rad/s

# Collect deformed meshes from each case
mesh_array = np.ones((len(dels),len(vels),np.size(mesh,axis=0),\
                      np.size(mesh,axis=1),np.size(mesh,axis=2)))
def_mesh_array = np.ones((len(dels),len(vels),np.size(mesh,axis=0),\
                      np.size(mesh,axis=1),np.size(mesh,axis=2)))
counter = 1
total = len(dels)*len(vels)

# Loop through deflections and velocities
for i,d in enumerate(dels):
    for j,v in enumerate(vels):
        print('Case ',counter,' of ',total)
        
        prob['delta_aileron'] = d
        prob['v'] = v
        prob['Mach_number'] = prob['v']/prob['speed_of_sound']
        
        # Recorder (can change so it doesn't overwrite itself)
        recorder = SqliteRecorder("aerostruct.db")
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['record_derivatives'] = True
        prob.driver.recording_options['includes'] = ['*']
        
        # Run
        prob.run_driver()
        
        mesh_array[i,j,:,:,:] = prob[point_name+'.coupled.'+name+'.mesh']
        def_mesh_array[i,j,:,:,:] = prob[point_name+'.coupled.'+name+'.def_mesh']
        p[i,j] = prob['omega'][0]
        Mach[j] = prob['Mach_number']
        
        counter += 1

print('Full analysis set complete!')



##############################################################################
#                     CALCULATE CONTROL EFFECTIVENESS                        #
##############################################################################
dels_rad = dels*np.pi/180        # Deflection angles, rad
p_deg = p*180/np.pi              # Angular velocity, deg/s
pl_U = p*(mesh_dict['span']/2)/vels  # roll rate * semispan / velocity

# Control effectiveness
CE = []
# For each velocity tested, CE = slope of linear fit the pl_U = f(d_ail)
for i in range(len(vels)):
    CE.append(np.polyfit(dels_rad,pl_U[:,i],1)[0])

#import matplotlib.pyplot as plt
##from mpl_toolkits.mplot3d import axes3d
#
#plt.figure(1,figsize=(9,4))
#plt.plot(q,CE,marker='s',color='k',linewidth=2)
#plt.xlabel('q (Pa)')
#plt.ylabel('Control Effectiveness')
##plt.savefig('control_effectiveness.png')
#plt.show()
#
#plt.figure(2,figsize=(9,4))
#D,Q = np.meshgrid(dels,q)
#plt.contourf(D,Q,p_deg)
#cbar = plt.colorbar()
#cbar.set_label('Roll rate (deg/s)')
#plt.xlabel('Aileron deflection angle (deg)')
#plt.ylabel('Dynamic pressure (Pa)')
##plt.savefig('roll_rate_contour.png')
#plt.show()