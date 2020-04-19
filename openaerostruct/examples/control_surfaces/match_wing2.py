from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder, \
                            ExecComp, NonlinearBlockGS, ExplicitComponent, n2
                            
from openaerostruct.utils.constants import grav_constant

from plot_tools import surfForcePlot, meshPlot

"""
    Takes a control surface deflection and gives the instantaneous moment.
    
    Overview:
        -Define mesh and control surface
        -Set up aerostructural point group as normal (rotational=False)
        -Run model
        -View moment
        
"""

num_y = 23
num_x = 5

##############################################################################
#                             SET UP GEOMETRY                                #
##############################################################################
# Based on NASA-TM-85674

# Thickness values
thickness_cp = np.array([0.1033,0.10833,0.11,0.1125,0.114,0.115,0.1175,0.127,0.1333,0.14,\
                     0.146,0.14,0.1333,0.127,0.1175,0.115,0.114,0.1125,0.11,0.10833,0.1033])
# Twist values
twist_cp = np.array([-2.5,-2.25,-2.,-1.88,-1.66,-1.5,-1.25,-0.25,0.55,1.,\
                     1.,1.,0.55,-0.25,-1.25,-1.5,-1.66,-1.88,-2.,-2.25,-2.5])

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : num_y,
             'num_x' : num_x,
             'wing_type' : 'rect',
             'symmetry' : False,
             'span_cos_spacing' : 1,
             'span' : 2*26.485*0.0254,
             'root_chord' : 10.5*0.0254,
             'offset' : np.array([10.5*0.0254/2, 0., 0.])}

mesh = generate_mesh(mesh_dict)

###### Fix the chords ######
ybr = 11.5*0.0254   # Break span position
cbr = 6.342*0.0254  # Break chord 

ctip = 3.07*0.0254      # Tip positoin
ytip = 26.485*0.0254    # Tip chord

croot = 10.5*0.0254 # Root position
yroot = 0           # Root chord

taper1 = np.polyfit([ytip,ybr],[ctip,cbr],1)    # Linearly change chords for
taper2 = np.polyfit([ybr,yroot],[cbr,croot],1)  # each of the sections

for i in range(num_y): # Alter the mesh
    chord_mesh = mesh[:,i,:]
    y = chord_mesh[0,1]
    
    if abs(y) >= ybr:
        c = taper1[0]*abs(y) + taper1[1]
    else:
        c = taper2[0]*abs(y) + taper2[1]
    
    chord_mesh[:,0] /= np.max(chord_mesh[:,0])
    chord_mesh[:,0] *= c
    mesh[:,i,:] = chord_mesh

# Define the control surfaces as in the paper
yvec = mesh[0,:,1]
yvec = -yvec[0:int((num_y-1)/2)]

# Can't set ailerons to have arbitrary span locations, need to match with
# existing panels so find the closest ones
ail1y = np.array([0.273,0.115])*ytip
ail2y = np.array([0.708,0.496])*ytip
ail3y = np.array([0.949,0.843])*ytip

yLoc1 = np.array([np.argmin(np.abs(yvec-ail1y[0])),np.argmin(np.abs(yvec-ail1y[1]))])
yLoc2 = np.array([np.argmin(np.abs(yvec-ail2y[0])),np.argmin(np.abs(yvec-ail2y[1]))])
yLoc3 = np.array([np.argmin(np.abs(yvec-ail3y[0])),np.argmin(np.abs(yvec-ail3y[1]))])

# Find errors from actual aileron sizes
yLoc1_err = yvec[yLoc1]-ail1y
yLoc2_err = yvec[yLoc2]-ail2y
yLoc3_err = yvec[yLoc3]-ail3y

ail1_err = 100*((yvec[yLoc1[0]]-yvec[yLoc1[1]])-(ail1y[0]-ail1y[1]))/(ail1y[0]-ail1y[1])
ail2_err = 100*((yvec[yLoc2[0]]-yvec[yLoc2[1]])-(ail2y[0]-ail2y[1]))/(ail2y[0]-ail2y[1])
ail3_err = 100*((yvec[yLoc3[0]]-yvec[yLoc3[1]])-(ail3y[0]-ail3y[1]))/(ail3y[0]-ail3y[1])

aileron1 = {
       'name': 'aileron1',
       'yLoc': [yLoc1[0],yLoc1[1]],       # Start and end chordwise mesh points for the surface
       'cLoc': [0.85,0.85],  # Local chord percentage of the hinge location
       'antisymmetric': True # Creates CS mirrored across XZ with opposite/equal deflections
       }

# For lower aileron2 errors, do num_y=23,span cos spacing = 1, error ~2%
aileron2 = {
       'name': 'aileron2',
       'yLoc': [yLoc2[0],yLoc2[1]],       # Start and end chordwise mesh points for the surface
       'cLoc': [0.8,0.8],  # Local chord percentage of the hinge location
       'antisymmetric': True # Creates CS mirrored across XZ with opposite/equal deflections
       }

# For lower aileron3 errors, do num_y=25,span cos spacing = 1, error ~6%
aileron3 = {
       'name': 'aileron3',
       'yLoc': [yLoc3[0],yLoc3[1]],       # Start and end chordwise mesh points for the surface
       'cLoc': [0.77,0.77],  # Local chord percentage of the hinge location
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
            
            'sweep' : 29.8,
            'twist_cp' : twist_cp,
            'thickness_cp' : thickness_cp,
            'dihedral' : 5.,
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
            'control_surfaces' : [aileron2]
            }


##############################################################################
#                                PARAMETERS                                  #
##############################################################################
CG_loc = np.array([9.76654*0.0254,0.,2.035*0.0254])
M_points = np.array([0.3,0.6,0.7,0.77,0.81,0.84,0.86])
Re = np.array([3.e6,5.e6]) # * (5.95/12) # lower Re only goes with lowest Mach
q = np.array([210,660,747,802,833,853,867])*47.8803 # Pa
dels = np.array([-10.,-7.5,-5.,-2.5,0.,2.5,5.,7.5,10.])
speed_of_sound = np.sqrt(1.4*287*322.039)
vels = M_points*speed_of_sound
rho = 2*q/(vels**2)
alphas = np.array([-6.,-4.,-2.,0.,2.,4.,6.,8.,10.])

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('v', val=248.136, units='m/s')
indep_var_comp.add_output('alpha', val=0.5, units='deg')
indep_var_comp.add_output('Mach_number', val=0.84)
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
indep_var_comp.add_output('CT', val=grav_constant * 17.e-6, units='1/s')
indep_var_comp.add_output('R', val=0., units='m')
indep_var_comp.add_output('W0', val=0.4 * 3e5,  units='kg')
indep_var_comp.add_output('speed_of_sound', val=295.4, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('cg', val=CG_loc, units='m')
indep_var_comp.add_output('delta_aileron', val=5.,units='deg')
indep_var_comp.add_output('omega', val=np.zeros((3)),units='rad/s')

prob.model.add_subsystem('prob_vars',
     indep_var_comp,
     promotes=['*'])

aerostruct_group = AerostructGeometry(surface=surface)

name = 'wing'

# Add tmp_group to the problem with the name of the surface.
prob.model.add_subsystem(name, aerostruct_group)

point_name = 'AS_point_0'

# Create the aero point group and add it to the model
AS_point = AerostructPoint(surfaces=[surface],compressible=True,rotational=True)

prob.model.add_subsystem(point_name, AS_point,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
        'W0', 'speed_of_sound', 'cg', 'load_factor', 'delta_aileron','omega'])

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

from openmdao.api import ScipyOptimizeDriver
prob.driver = ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-9

#recorder = SqliteRecorder("roll_run.db")
#prob.driver.add_recorder(recorder)
#prob.driver.recording_options['record_derivatives'] = True
#prob.driver.recording_options['includes'] = ['*']


# Set up the problem
prob.setup(check=True)

# View the model
# n2(prob)

# Setup data storage
meshMat = np.zeros((len(M_points),len(alphas),len(dels),num_x,num_y,3))
meshForce = np.zeros((len(M_points),len(alphas),len(dels),num_x,num_y,3))
panelForce = np.zeros((len(M_points),len(alphas),len(dels),num_x-1,num_y-1,3))
p = np.zeros((len(M_points),(len(alphas)),(len(dels))))
CL = np.zeros((len(M_points),(len(alphas)),(len(dels))))
CD = np.zeros((len(M_points),(len(alphas)),(len(dels))))
CM = np.zeros((len(M_points),len(alphas),len(dels),3))

prob['cg'] = CG_loc
for i in range(len(M_points)):
    for j in range(len(alphas)):
        for k in range(len(dels)):
            prob['Mach_number'] = M_points[i]
            if M_points[i]==0.3:
                prob['re'] = 3.e6
            else:
                prob['re'] = 5.e6
            prob['v'] = vels[i]
            prob['rho'] = rho[i]
            prob['alpha'] = alphas[j]
            prob['speed_of_sound'] = speed_of_sound
            prob['delta_aileron'] = dels[k]
            
            # Run the model
            prob.run_driver()
            
            CL[i,j,k] = prob['AS_point_0.wing_perf.CL'][0]
            CD[i,j,k] = prob['AS_point_0.wing_perf.CD'][0]
            CM[i,j,k,:] = prob['AS_point_0.total_perf.moment.CM'];
            meshMat[i,j,k,:,:,:] = prob['AS_point_0.coupled.wing.def_mesh']
            meshForce[i,j,k,:,:,:] = prob['AS_point_0.coupled.aero_states.wing_mesh_point_forces']
            panelForce[i,j,k,:,:,:] = prob['AS_point_0.total_perf.wing_sec_forces']
            p[i,j,k] = prob['omega'][0]

print('Completed!\n')
np.save('steadyRoll_M',M_points)
np.save('steadyRoll_rho',rho)
np.save('steadyRoll_dels',dels)
np.save('steadyRoll_alpha',alphas)
np.save('steadyRoll_V',vels)
np.save('steadyRoll_a',speed_of_sound)
np.save('steadyRoll_M',M_points)
np.save('steadyRoll_CL',CL)
np.save('steadyRoll_CD',CD)
np.save('steadyRoll_CM',CM)
np.save('steadyRoll_omega',p)
np.save('steadyRoll_defmesh',meshMat)
np.save('steadyRoll_meshF',meshForce)
np.save('steadyRoll_panelF',panelForce)

dels_rad = dels*np.pi/180        # Deflection angles, rad
p_deg = p*180/np.pi              # Angular velocity, deg/s
#pl_U = p*(mesh_dict['span']/2)/vels  # roll rate * semispan / velocity

# Control effectiveness
#CE = []
# For each velocity tested, CE = slope of linear fit the pl_U = f(d_ail)
#for i in range(len(vels)):
#    CE.append(np.polyfit(dels_rad,pl_U[:,i],1)[0])