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

num_y = 47
num_x = 5

##############################################################################
#                             SET UP GEOMETRY                                #
##############################################################################
# Based on NASA-TM-85674

# Thickness values
t_over_c_cp = np.array([0.1033,0.10833,0.11,0.1125,0.114,0.115,0.1175,0.127,0.1333,0.14,\
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

##############################################################################
#                        ADJUST CHORDS and CAMBER                            #
##############################################################################
z_x_data = np.array([[0.014733221650630624, -0.0034824247994900137],
                [0.055027945384193255, -0.0029380457503743324],
                [0.10107333995522794, -0.002395445717759029],
                [0.14518850162335245, -0.002037863401183046],
                [0.19121388225875774, -0.0017736794508768985],
                [0.22958504440128674, -0.0013215127570308308],
                [0.27561042503669225, -0.0010573288067246764],
                [0.33313714734704153, -0.0007967028894192985],
                [0.3849332127555482, -0.00025588187330435885],
                [0.4751471394897188, 0.0013867433620446887],
                [0.5289334796080234, 0.0029478303411264173],
                [0.586606970779654, 0.005250174195365671],
                [0.627001764191363, 0.007186633656027154],
                [0.6674232428505775, 0.009494314559767542],
                [0.715518953938298, 0.011892428802277137],
                [0.7674150890249509, 0.013825330229937885],
                [0.8144389426711933, 0.014646049842112284],
                [0.8575556313284807, 0.014447096496820008],
                [0.8919662579870431, 0.01313714734704164],
                [0.9244066238714366, 0.01108534831660564],
                [0.9472091678650321, 0.00829407142751249],
                [0.9709034438794422, 0.004574444427972071],
                [0.9869768579603577, 0.0015068269758201983],
                [1.0021185121492004, -0.0011892725305027163],
                [1.011542852059953, -0.003419566216476655]])

z_fit = np.polyfit(z_x_data[:,0],z_x_data[:,1],7)

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
    
    # Apply the camber points
    chord_mesh[:,2] = np.polyval(z_fit,chord_mesh[:,0])
    chord_mesh[:,0] *= c
    chord_mesh[:,2] *= c
    mesh[:,i,:] = chord_mesh


S_paper = 0.18469124 # m^2, trapazoidal reference area given in paper

##############################################################################
#                         DEFINE CONTROL SURFACES                            #
##############################################################################
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
       'antisymmetric': False # Creates CS mirrored across XZ with opposite/equal deflections
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
            'S_ref_type' : 'projected', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',
            
            'sweep' : 30,
            'twist_cp' : twist_cp,
            'thickness_cp' : np.ones((3))*.0005,
            'radius_cp' : np.array([3.07*0.0254*0.11, 10.5*0.0254*0.14]),
            'dihedral' : 5.,
            'mesh' : mesh,
            #'taper' : 0.29238095,
            
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
            't_over_c_cp' : t_over_c_cp,      # thickness over chord ratio (NACA0015)
            'c_max_t' : .31,       # chordwise location of maximum (NACA0015)
                                    # thickness
            'with_viscous' : True,
            'with_wave' : False,     # if true, compute wave drag

#            # Structural values are based on aluminum 7075
#            'E' : 70.e9,            # [Pa] Young's modulus of the spar
#            'G' : 30.e9,            # [Pa] shear modulus of the spar
#            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
#            'mrho' : 3.e3,          # [kg/m^3] material density
#            'fem_origin' : 0.35,    # normalized chordwise location of the spar
#            'wing_weight_ratio' : 2.,
#            'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
#            'distributed_fuel_weight' : False,
            
#            # Structural values are based on other aluminum 7075
#            'E' : 71.e9,            # [Pa] Young's modulus of the spar
#            'G' : 26.9.e9,            # [Pa] shear modulus of the spar
#            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
#            'mrho' : 3.e3,          # [kg/m^3] material density
#            'fem_origin' : 0.35,    # normalized chordwise location of the spar
#            'wing_weight_ratio' : 2.,
#            'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
#            'distributed_fuel_weight' : False,
#            
            # Structural values are based on steel
            'E' : 200.e9,            # [Pa] Young's modulus of the spar
            'G' : 75.e9,            # [Pa] shear modulus of the spar
            'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 8.e3,          # [kg/m^3] material density
            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            'wing_weight_ratio' : 1.25,
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
Re = np.array([3.e6,5.e6]) * (5.95/12) # lower Re only goes with lowest Mach
q = np.array([210,660,747,802,833,853,867])*47.8803 # Pa
dels = np.array([-10.,-5.,0.,5.,10.])
speed_of_sound = np.sqrt(1.4*287*322.039)
vels = M_points*speed_of_sound
rho = 2*q/(vels**2)
alphas = np.array([-4.,-2.,0.,2.,4.,6.,8.,10.,12.,14.])

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

from openmdao.api import ScipyOptimizeDriver
prob.driver = ScipyOptimizeDriver()
prob.driver.options['tol'] = 1e-9

#recorder = SqliteRecorder("roll_run.db")
#prob.driver.add_recorder(recorder)
#prob.driver.recording_options['record_derivatives'] = True
#prob.driver.recording_options['includes'] = ['*']



##############################################################################
#                          SET UP AND RUN MODEL                              #
##############################################################################
# Set up the problem
prob.setup(check=True)

# View the model
# n2(prob)

# Setup data storage
# Setup data storage
meshMat = np.zeros((len(M_points),len(alphas),len(dels),num_x,num_y,3))
meshForce = np.zeros((len(M_points),len(alphas),len(dels),num_x,num_y,3))
panelForce = np.zeros((len(M_points),len(alphas),len(dels),num_x-1,num_y-1,3))
CL = np.zeros((len(M_points),(len(alphas)),(len(dels))))
CD = np.zeros((len(M_points),(len(alphas)),(len(dels))))
CM = np.zeros((len(M_points),len(alphas),len(dels),3))
cl = np.zeros((len(M_points),len(alphas),len(dels))) # Roll moment coefficient
corrector = np.zeros((len(M_points),len(alphas),len(dels)))

import pdb
prob['cg'] = CG_loc
#for i in range(len(M_points)):
for i in range(1):
    for j in range(len(alphas)):
        for k in range(len(dels)):
            prob['Mach_number'] = M_points[i]
            if M_points[i]==0.3:
                prob['re'] = 3.e6 * (5.95/12)
            else:
                prob['re'] = 5.e6 * (5.95/12)
            prob['v'] = vels[i]
            prob['rho'] = rho[i]
            prob['alpha'] = alphas[j]
            prob['speed_of_sound'] = speed_of_sound
            prob['delta_aileron'] = dels[k]
            
            # Run the model
            prob.run_model()
            
            CL[i,j,k] = prob['AS_point_0.wing_perf.CL'][0]
            CD[i,j,k] = prob['AS_point_0.wing_perf.CD'][0]
            CM[i,j,k,:] = prob['AS_point_0.total_perf.moment.CM'];
            meshMat[i,j,k,:,:,:] = prob['AS_point_0.coupled.wing.def_mesh']
            meshForce[i,j,k,:,:,:] = prob['AS_point_0.coupled.aero_states.wing_mesh_point_forces']
            panelForce[i,j,k,:,:,:] = prob['AS_point_0.total_perf.wing_sec_forces']
            
            
            # Add a corrector for the coefficients to make sure the 
            # normalization factor matches that of the paper
            # cl = roll moment/S_paper*b, but we have CM_x = roll moment/S*MAC
            S_ref = prob['AS_point_0.coupled.wing.S_ref']
            chords = prob['AS_point_0.coupled.wing.chords']
            widths = prob['AS_point_0.coupled.wing.widths']
            panel_chords = (chords[1:] + chords[:-1]) * 0.5
            MAC = 1. / S_ref * np.sum(panel_chords**2 * widths)
            S = prob['AS_point_0.total_perf.S_ref_total']
#                                      S_paper from trapazoidal reference wing
            corrector[i,j,k] = (S*MAC)/(S_paper * mesh_dict['span'])
            cl[i,j,k] = corrector[i,j,k]*CM[i,j,k,0]
            #pdb.set_trace()
            
print('Completed!\n')

M = M_points[0]
    
import matplotlib.pyplot as plt

plt.figure(1,figsize=(11,5))
plt.plot(alphas,np.reshape(-cl[0,:,0],(10,)),color='k',marker='P')
plt.plot(alphas,np.reshape(-cl[0,:,1],(10,)),color='k',marker='D')
plt.plot(alphas,np.reshape(-cl[0,:,2],(10,)),color='k',marker='o')
plt.plot(alphas,np.reshape(-cl[0,:,3],(10,)),color='k',marker='s')
plt.plot(alphas,np.reshape(-cl[0,:,4],(10,)),color='k',marker='^')
plt.xlim([-6.,16.])
plt.ylim([-0.01,0.04])
plt.xlabel(r'$\alpha$ (deg)')
plt.ylabel(r'$c_l$')
plt.legend(['-10 deg','-5 deg','0 deg','5 deg','10 deg'])
plt.title('M='+str(M))
plt.grid()
plt.savefig('alpha_v_cl_M'+str(M)+'.png')
plt.show()
        
np.save('rollMom_M',M_points)
np.save('rollMom_rho',rho)
np.save('rollMom_dels',dels)
np.save('rollMom_alpha',alphas)
np.save('rollMom_V',vels)
np.save('rollMom_a',speed_of_sound)
np.save('rollMom_CL',CL)
np.save('rollMom_CD',CD)
np.save('rollMom_CM',CM)
np.save('rollMom_Cl',cl)
np.save('rollMom_defmesh',meshMat)
np.save('rollMom_meshF',meshForce)
np.save('rollMom_panelF',panelForce)