from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder, NewtonSolver,\
                            ExecComp, NonlinearBlockGS, ExplicitComponent, n2
                            
from openaerostruct.utils.constants import grav_constant

#from plotting import surfForcePlot, meshPlot

import pdb

"""
    Takes a control surface deflection and gives the instantaneous moment.
    
    Overview:
        -Define mesh and control surface
        -Set up aerostructural point group as normal (rotational=False)
        -Run model
        -View moment
        
"""

##############################################################################
#                                  GEOMETRY                                  #
##############################################################################
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
dels = np.array([-10.,-5.,0.,5.,10.])
qvec = np.array([10,15,20,25,30,35,40,45,50,55,60]) * 47.8803 # maybe ad 0.07 = q too
alpha = 0.012 # From Kirsten Wind Tunnel page
rho = 1.2
vels = np.sqrt(2*qvec/rho)

n = 3
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

##############################################################################
#                                 SPAR SETUP                                 #
##############################################################################
yvec = mesh[0,:,1]
ys = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) * b/2

# For straight wing
GJ_straight_data = np.array([0.006335115864527629,
                       0.005493761140819965,
                       0.00398217468805704,
                       0.002713012477718357,
                       0.0017718360071301217,
                       0.0011158645276292314,
                       0.0007165775401069465,
                       0.00044563279857396873,
                       0.00028877005347592924,
                       0.00020320855614972683,
                       0.00018894830659535976])*0.00286981466*1e-3 # Going from lb-in^2 to N-m^2

EI_straight_data = np.array([0.007247771836007131,
                             0.006221033868092692,
                             0.0044242424242424225,
                             0.0030124777183600706,
                             0.0019999999999999983,
                             0.001286987522281638,
                             0.0008163992869875195,
                             0.0005169340463458075,
                             0.0003458110516933975,
                             0.00026024955436719857,
                             0.0002459893048128263])*0.00286981466*1e-3

GJ_straight_fit = np.polyfit(ys,GJ_straight_data,3)
GJ_straight = np.polyval(GJ_straight_fit,np.abs(yvec))

EI_straight_fit = np.polyfit(ys,EI_straight_data,3)
EI_straight = np.polyval(EI_straight_fit,np.abs(yvec))

G = 50.e9
spar_radius_straight = np.array([0.18*0.0254,0.28*0.0254,0.18*0.0254])

t = np.linspace(0.18*0.0254,0.28*0.0254,int((num_y-1)/2)+1)
t = np.hstack((t,np.flip(t)[1:]))

Gs = (2*GJ_straight)/(np.pi * t**4)
spar_thickness_straight = spar_radius_straight

#G_str = np.sum((2*GJ_straight)/(np.pi * spar_radius_straight**4))/len(yvec)



GJ_swept_data = np.array([0.010014260249554367,
                     0.00863101604278075,
                     0.006206773618538325,
                     0.004281639928698752,
                     0.0028413547237076623,
                     0.0017860962566844905,
                     0.0010873440285204955,
                     0.0006452762923351146,
                     0.00041711229946523633,
                     0.00033155080213903045,
                     0.0003172905525846599])*0.00286981466*1e-3 # Going from lb-in^2 to N-m^2
    
EI_swept_data = np.array([0.011454545454545457,
                          0.009843137254901962,
                          0.007062388591800359,
                          0.004852049910873438,
                          0.0032406417112299454,
                          0.0020570409982174666,
                          0.0012584670231729003,
                          0.0007450980392156824,
                          0.00047415329768270287,
                          0.0003743315508021386,
                          0.00040285204991086926])*0.00286981466*1e-3

GJ_swept_fit = np.polyfit(ys,GJ_swept_data,3)
GJ_swept = np.polyval(GJ_swept_fit,np.abs(yvec))

EI_swept_fit = np.polyfit(ys,EI_swept_data,3)
EI_swept = np.polyval(EI_swept_fit,np.abs(yvec))

spar_radius_swept = ((2*GJ_swept)/(np.pi*G))**(1/4)
spar_thickness_swept = spar_radius_swept

E_swp = np.sum((4*EI_swept)/(np.pi * spar_radius_swept**4))/len(yvec)


spar_b = np.array([0.002,0.0038,0.002])
spar_a = np.array([0.00088,0.003,0.00088])
E_swp = 125.e9


##############################################################################
#                              CONTROL SURFACES                              #
##############################################################################
# All ailerons are 0.3c
c = [0.7,0.7]

# Aileron locations for straight wing
ind02 = [0,n]
ind04 = [0,2*n]
ind06 = [0,3*n]
ind08 = [0,4*n]
ind1 =  [0,5*n]

# Inboard aileron locations for swept wing
ind02in = [n,2*n]
ind04in = [n,3*n]

ail02 = {
       'name': 'ail02',
       'yLoc': ind02,
       'cLoc': c,
       'antisymmetric': True 
       }
ail04 = {
       'name': 'ail04',
       'yLoc': ind04,
       'cLoc': c,
       'antisymmetric': True 
       }
ail06 = {
       'name': 'ail06',
       'yLoc': ind06,
       'cLoc': c,
       'antisymmetric': True 
       }
ail08 = {
       'name': 'ail08',
       'yLoc': ind08,
       'cLoc': c,
       'antisymmetric': True 
       }
ail1 = {
       'name': 'ail1',
       'yLoc': ind1,
       'cLoc': c,
       'antisymmetric': True 
       }

ail02in = {
       'name': 'ail02in',
       'yLoc': ind02in,
       'cLoc': c,
       'antisymmetric': True 
       }
ail04in = {
       'name': 'ail04in',
       'yLoc': ind04in,
       'cLoc': c,
       'antisymmetric': True 
       }



##############################################################################
#                                 SURFACES                                   #
##############################################################################
straight_wing = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : False,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'projected', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
                                     
            'fem_model_type' : 'tube',
            #'tube_Xsec' : 'rect',       # rectangular Xsec instead of usual tube
            'thickness_cp' : np.array([0.18*0.0254,0.24*0.0254,0.18*0.0254]),
            'radius_cp' : np.array([0.18*0.0254,0.24*0.0254,0.18*0.0254]),
            'twist' : np.array([0.]),
            'sweep' : np.array([0.]),
            'mesh' : mesh,
            'taper' : taper,
            
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

            # Structural values are based on beryllium copper
            'E' : 24.5e9,            # [Pa] Young's modulus of the spar
            'G' : 5.5e9,            # [Pa] shear modulus of the spar
            'yield' : 1280.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 8.25e3,          # [kg/m^3] material density
            
            'fem_origin' : 0.38,    # normalized chordwise location of the spar, 0.38c
            'wing_weight_ratio' : 1.5,
            'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
            'distributed_fuel_weight' : False,
            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            }

swept_wing = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : False,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'projected', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
                                     
            'fem_model_type' : 'tube',
            'thickness_cp' : spar_thickness_swept,
            'radius_cp' : spar_radius_swept,
            'twist' : np.array([0.]),
            'sweep' :  sweep,
            'mesh' : mesh,
            'taper' : taper,
            
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

            # Structural values are based on beryllium copper
            'E' : 24.5e9,            # [Pa] Young's modulus of the spar
            'G' : G,            # [Pa] shear modulus of the spar
            'yield' : 1280.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
            'mrho' : 8.25e3,          # [kg/m^3] material density
            
            'fem_origin' : 0.38,    # normalized chordwise location of the spar, 0.38c
            'wing_weight_ratio' : 1.5,
            'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
            'distributed_fuel_weight' : False,
            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            }

surfList = [straight_wing,swept_wing]
ailList_straight = [ail02,ail04,ail06,ail08,ail1]
ailList_swept = [ail02,ail04,ail06,ail08,ail1,ail02in,ail04in]

counter = 0
for surface in surfList:
    if surface == straight_wing:
        ailList = ailList_straight
    else:
        ailList = ailList_swept
    
    for aileron in ailList:
        surface['control_surfaces'] = [aileron]
        
        cl = np.ones((len(vels),len(dels)))
        CL = np.ones((len(vels),len(dels)))
        CD = np.ones((len(vels),len(dels)))
        CM = np.ones((len(vels),len(dels),3))
        defmeshes = np.zeros((len(vels),len(dels),num_x,num_y,3))
        meshForce = np.zeros((len(vels),len(dels),num_x,num_y,3))
        panelForce = np.zeros((len(vels),len(dels),num_x-1,num_y-1,3))
        ###################################################################
        #                          SET UP PROBLEM                         #
        ###################################################################
        # Create the problem and assign the model group
        prob = Problem()
                
        # Add problem information as an independent variables component                
        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v', val=25., units='m/s')
        indep_var_comp.add_output('alpha', val=alpha, units='deg')
        indep_var_comp.add_output('Mach_number', val=25./340.5)
        indep_var_comp.add_output('re', val=5e5, units='1/m')
        indep_var_comp.add_output('rho', val=rho, units='kg/m**3')
        indep_var_comp.add_output('CT', val=grav_constant * 17.e-6, units='1/s')
        indep_var_comp.add_output('R', val=0., units='m')
        indep_var_comp.add_output('W0', val=5.,  units='kg')
        indep_var_comp.add_output('speed_of_sound', val=340.5, units='m/s')
        indep_var_comp.add_output('load_factor', val=1.)
        indep_var_comp.add_output('cg', val=cg_loc, units='m')
        indep_var_comp.add_output('delta_aileron', val=12.5,units='deg')
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
        AS_point = AerostructPoint(surfaces=[surface],rotational=True,compressible=False)
        
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
        prob.model.connect(name + '.structural_mass', point_name + '.' + 'total_perf.' + name + '_structural_mass')
        prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')
        
        from openmdao.api import ScipyOptimizeDriver
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['tol'] = 1e-9
            
        # Set up the problem
        prob.setup(check=True)
        
        prob.model.AS_point_0.coupled.nonlinear_solver.options['maxiter'] = 750
        prob.model.AS_point_0.coupled.nonlinear_solver.options['err_on_maxiter'] = False
        prob.model.AS_point_0.coupled.nonlinear_solver.options['atol'] = 5e-7
        ###################################################################
        #                            RUN PROBLEM                          #
        ###################################################################
        
        for i,v in enumerate(vels):
            for j,d in enumerate(dels):
                prob['v'] = v
                prob['Mach_number'] = v/340.5
                prob['delta_aileron'] = d
                prob['re'] = (rho*6.5*0.0254*v)/(1.4207e-5)
                prob['load_factor'] = 1.
                
                print('v='+str(v))
                print('d='+str(d))
                # Run the model
                prob.run_model()

                # get cl instead of CM
                S_ref = prob['AS_point_0.coupled.wing.S_ref']
                chords = prob['AS_point_0.coupled.wing.chords']
                widths = prob['AS_point_0.coupled.wing.widths']
                panel_chords = (chords[1:] + chords[:-1]) * 0.5
                MAC = 1. / S_ref * np.sum(panel_chords**2 * widths)
                
                defmeshes[i,j,:,:,:] = prob['AS_point_0.coupled.wing.def_mesh']
                CL[i,j] = prob['AS_point_0.wing_perf.CL'][0]
                CM[i,j,:] = prob['AS_point_0.total_perf.moment.CM'];
                cl[i,j] = MAC * CM[i,j,0]/b
                meshForce[i,j,:,:,:] = prob['AS_point_0.coupled.aero_states.wing_mesh_point_forces']
                panelForce[i,j,:,:,:] = prob['AS_point_0.total_perf.wing_sec_forces']
                #pdb.set_trace()
            
        cl_del = np.zeros(len(vels))
        for i in range(len(vels)):
            cl_del[i] = np.polyfit(dels,cl[i,:],1)[0]
            
        np.save('case_'+str(counter)+'cl_del',cl_del)
        np.save('case_'+str(counter)+'cl',cl)
        np.save('case_'+str(counter)+'CL',CL)
        np.save('case_'+str(counter)+'CM',CM)
        np.save('case_'+str(counter)+'defmesh',defmeshes)
        counter+=1
        pdb.set_trace()
                
print()
print('Completed!')
print()