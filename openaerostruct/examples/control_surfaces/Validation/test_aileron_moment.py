from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder, NewtonSolver,\
                            ExecComp, NonlinearBlockGS, ExplicitComponent, n2
                            
from openaerostruct.utils.constants import grav_constant

from data_TN2563_fig18_19 import dcl_ddelta_data

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


###############################################################################
#                                   SPAR                                      #
###############################################################################                     0 b/2    0.25 b/2        0.5 b/2     0.75 b/2      1 b/2
ys = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
yv = np.linspace(0,1,25)
struc = np.array([1.3,1.27,1.2,1.135,1.09,1.05,1.03,1.02,1.011,1,1])
data_swp = np.array([31.56179775, 27.20224719, 19.56179775, 13.49438202,  8.95505618,
        5.62921348,  3.42696629,  2.03370787,  1.31460674,  1.04494382, 1.])
fit_swp = np.polyfit(ys,data_swp,3)
j_swp = np.polyval(fit_swp,yv)
r_swp = (2*j_swp/np.pi)**(1/4)
r_swp = np.hstack((np.flip(r_swp)[:-1],r_swp))

data_str = np.array([28.43362832, 24.43362832, 17.53097345, 12.04424779,  8.04424779,
        5.10619469,  3.12389381,  1.84955752,  1.17699115,  0.92920354, 1.])
fit_str = np.polyfit(ys,data_str,3)
j_str = np.polyval(fit_str,yv)
r_str = (2*j_str/np.pi)**(1/4)
r_str = np.hstack((np.flip(r_str)[:-1],r_str))

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
ind95 = [0,ind1[1]-1]

# Inboard aileron locations for swept wing
ind02in = [n,2*n]
ind04in = [n,3*n]

ail02 = {
       'name': 'ail02',
       'yLoc': ind02,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : True
       }
ail04 = {
       'name': 'ail04',
       'yLoc': ind04,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : True
       }
ail06 = {
       'name': 'ail06',
       'yLoc': ind06,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : False
       }
ail08 = {
       'name': 'ail08',
       'yLoc': ind08,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : False
       }
ail1 = {
       'name': 'ail1',
       'yLoc': ind1,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : False
       }
ail95 = {
       'name': 'ail95',
       'yLoc': ind95,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : False
       }
ail02in = {
       'name': 'ail02in',
       'yLoc': ind02in,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : True
       }
ail04in = {
       'name': 'ail04in',
       'yLoc': ind04in,
       'cLoc': c,
       'antisymmetric': True,
       'corrector' : True
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
                                     
            'fem_model_type' : 'tube',# b2 .25b2 .5b2  .75b2
            'thickness_cp' : r_str*0.155*0.0254,
            'radius_cp' : r_str*0.155*0.0254,
            'twist' : np.array([0.]),
            'sweep' : 0,
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

            'E' : 8.7e9,            # [Pa] Young's modulus of the spar
            'G' : 2.82e9,            # [Pa] shear modulus of the spar
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
            'thickness_cp' : r_swp*0.146*0.0254,
            'radius_cp' : r_swp*0.146*0.0254,
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

            'E' : 25.7e9,            # [Pa] Young's modulus of the spar
            'G' : 4.12e9,            # [Pa] shear modulus of the spar
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
ailList_straight = [ail02,ail04,ail06,ail08,ail95]
ailList_swept = [ail02,ail04,ail06,ail08,ail95]

for surface in surfList:
    if surface['sweep'] == 0:
        surfname = 'str'
        ailList = ailList_straight
    else:
        surfname = 'swp'
        ailList = ailList_swept
    
    for aileron in ailList:
        surface['control_surfaces'] = [aileron]
        print(surfname+' '+aileron['name']+'\n')
        ###################################################################
        #                          SET UP PROBLEM                         #
        ###################################################################
        # Create the problem and assign the model group
        prob = Problem()
                
        # Add problem information as an independent variables component                
        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v', val=25., units='m/s')
        indep_var_comp.add_output('alpha', val=alpha, units='deg')
        indep_var_comp.add_output('re', val=5e5, units='1/m')
        indep_var_comp.add_output('rho', val=rho, units='kg/m**3')
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

        ##from openmdao.api import ScipyOptimizeDriver
        #prob.driver = ScipyOptimizeDriver()
        #prob.driver.options['tol'] = 1e-9

        # Set up the problem
        prob.setup(check=True)
        prob.model.AS_point_0.coupled.nonlinear_solver.options['maxiter'] = 10000
        prob.model.AS_point_0.coupled.nonlinear_solver.options['atol'] = 1e-6
        prob.model.AS_point_0.coupled.nonlinear_solver.options['err_on_non_converge'] = True

        ###################################################################
        #                            RUN PROBLEM                          #
        ###################################################################

        for q_psf, cl_del_true in dcl_ddelta_data[surfname][aileron['name']]: # iterate lines
            q_pa = q_psf*47.8803
            v = np.sqrt(2*q_pa/rho)
            d = 5. #aileron deflection in deg

            prob['v'] = v
            prob['delta_aileron'] = d
            prob['re'] = (rho*6.5*0.0254*v)/(1.4207e-5)

            print('v='+str(v))
            print('d='+str(d))
            # Run the model
            prob.run_model()

            # calculate MAC
            S_ref = prob['AS_point_0.coupled.wing.S_ref'][0]
            chords = prob['AS_point_0.coupled.wing.chords']
            widths = prob['AS_point_0.coupled.wing.widths']
            panel_chords = (chords[1:] + chords[:-1]) * 0.5
            MAC = 1. / S_ref * np.sum(panel_chords**2 * widths)

            # get cl instead of CM
            CM = prob['AS_point_0.total_perf.moment.CM']
            cl = MAC * CM[0]/b
            cl_del = -cl/d # change sign to be consistent with NACA TN
            print(q_psf, cl)

            # Get the derivative dcl_ddelta by fitting a line to the cl(delta) data
            # obtained above

            print('=============================================================')
            print(surfname, aileron['name'], q_psf)
            print(cl_del,cl_del_true,abs(cl_del-cl_del_true))
            assert(abs(cl_del-cl_del_true)<6e-4)  # resolution of the report data is ~1e-4
