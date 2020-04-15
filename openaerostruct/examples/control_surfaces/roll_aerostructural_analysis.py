from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openmdao.api import IndepVarComp, Problem, Group, SqliteRecorder, \
                            ExecComp, NonlinearBlockGS, ExplicitComponent, n2
                            
from openaerostruct.utils.constants import grav_constant


"""
    Takes a control surface deflection and gives the instantaneous moment.
    
    Overview:
        -Define mesh and control surface
        -Set up aerostructural point group as normal (rotational=False)
        -Run model
        -View moment
        
"""

# Create a dictionary to store options about the surface
mesh_dict = {'num_y' : 21,
             'num_x' : 9,
             'wing_type' : 'CRM',
             'symmetry' : False,
             'num_twist_cp' : 5}

mesh, twist_cp = generate_mesh(mesh_dict)

# Create a dictionary for the control surface
aileron = {
       'name': 'aileron',
       'yLoc': [1,4],       # Start and end chordwise mesh points for the surface
       'cLoc': [0.75,0.75],  # Local chord percentage of the hinge location
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

            'thickness_cp' : np.array([.1, .2, .3]),

            'twist_cp' : twist_cp,
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
indep_var_comp.add_output('R', val=11.165e6, units='m')
indep_var_comp.add_output('W0', val=0.4 * 3e5,  units='kg')
indep_var_comp.add_output('speed_of_sound', val=295.4, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('empty_cg', val=np.zeros((3)), units='m')
indep_var_comp.add_output('delta_aileron', val=12.5,units='deg')

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

recorder = SqliteRecorder("roll_run.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']


# Set up the problem
prob.setup(check=True)

# View the model
# n2(prob)

# Run the model
prob.run_model()

print()
print('CL:', prob['AS_point_0.wing_perf.CL'])
print('CD:', prob['AS_point_0.wing_perf.CD'])
print('CM:', prob['AS_point_0.total_perf.moment.CM'])


def actualPlots(problem):
    from mpl_toolkits.mplot3d import Axes3D  
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import pdb
    
    mesh = prob[point_name+'.coupled.'+name+'.mesh']
    def_mesh = prob[point_name+'.coupled.'+name+'.def_mesh']
    forces = prob['AS_point_0.coupled.aero_states.wing_mesh_point_forces']
    
    Fx = forces[:,:,0]
    Fy = forces[:,:,1]
    Fz = forces[:,:,2]
    
    fx = Fx/Fx.max()
    fy = Fy/Fy.max()
    fz = Fz/Fz.max()
    
    #X1 = mesh[:,:,0];
    #Y1 = mesh[:,:,1];
    #Z1 = mesh[:,:,2];
    
    X2 = def_mesh[:,:,0];
    Y2 = def_mesh[:,:,1];
    Z2 = def_mesh[:,:,2];
    
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X2.max()-X2.min(), Y2.max()-Y2.min(), Z2.max()-Z2.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X2.max()+X2.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y2.max()+Y2.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z2.max()+Z2.min())    
       
    # Plot X forces
    fig1 = plt.figure(1,figsize=(5,4),dpi=300)
    ax1 = fig1.gca(projection='3d')
    
    surf1 = ax1.plot_surface(
        X2, Y2, Z2, rstride=1, cstride=1,
        facecolors=cm.coolwarm(fx),
        linewidth=0, antialiased=False, shade=False)
    
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax1.plot([xb], [yb], [zb], 'w')
    ax1.plot_wireframe(X2,Y2,Z2,color='k',linewidth=0.25)
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.elev=90.
    ax1.azim=255.
    ax1.set_zlim([-2.,2.])
    plt.grid()
    plt.savefig('surf_fx.png')
    plt.show()
    
    # Plot Y forces
    fig2 = plt.figure(1,figsize=(5,4),dpi=300)
    ax2 = fig2.gca(projection='3d')
    
    surf2 = ax2.plot_surface(
        X2, Y2, Z2, rstride=1, cstride=1,
        facecolors=cm.coolwarm(fy),
        linewidth=0, antialiased=False, shade=False)
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax2.plot([xb], [yb], [zb], 'w')
    ax2.plot_wireframe(X2,Y2,Z2,color='k',linewidth=0.25)

    fig2.colorbar(surf1, shrink=0.5, aspect=5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.elev=25.
    ax2.azim=255.
    ax2.set_zlim([-2.,2.])
    plt.grid()
    plt.savefig('surf_fy.png')
    plt.show()
    
    # Plot Z forces
    fig3 = plt.figure(1,figsize=(5,4),dpi=300)
    ax3 = fig3.gca(projection='3d')
    
    surf3 = ax3.plot_surface(
        X2, Y2, Z2, rstride=1, cstride=1,
        facecolors=cm.coolwarm(fz),
        linewidth=0, antialiased=False, shade=False)
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax3.plot([xb], [yb], [zb], 'w')
    ax3.plot_wireframe(X2,Y2,Z2,color='k',linewidth=0.25)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.elev=25.
    ax3.azim=255.
    ax3.set_zlim([-2.,2.])
    fig3.colorbar(surf1, shrink=0.5, aspect=5)


    plt.grid()
    plt.savefig('surf_fz.png')
    plt.show()

actualPlots(prob)