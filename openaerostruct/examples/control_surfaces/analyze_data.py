import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as manimation
import pdb
from plot_tools import meshPlot

M_points = np.load('rollMom_M.npy')
rho = np.load('rollMom_rho.npy')
dels = np.load('rollMom_dels.npy')
alphas = np.load('rollMom_alpha.npy')
vels = np.load('rollMom_V.npy')
speed_of_sound = np.load('rollMom_a.npy')
CL = np.load('rollMom_CL.npy')
CD = np.load('rollMom_CD.npy')
CM = np.load('rollMom_CM.npy')
cl = np.load('rollMom_cl.npy')
meshMat = np.load('rollMom_defmesh.npy')
meshForce = np.load('rollMom_meshF.npy')
panelForce = np.load('rollMom_panelF.npy')

# For ails on right wing
# pos delta = neg moment = AIL DOWN
# neg delta = pos moment = AIL UP

# For left wing
# neg delta = pos moment = AIL DOWN
# pos delta = neg moment = AIL UP

# Plot angle of attack vs. roll moment coefficient
# dels correlate to
# -10   = 10 deg down -> +10
# -5    = 5  deg down -> +5
# 0     = 0  deg down -> 0
# 5     = 5  deg up   -> -5
# 10    = 10 deg up   -> -10
for i,M in enumerate(M_points):
    
    plt.figure(1,figsize=(11,5))
    plt.plot(alphas,np.reshape(-cl[i,:,0],(10,)),color='k',marker='P')
    plt.plot(alphas,np.reshape(-cl[i,:,1],(10,)),color='k',marker='D')
    plt.plot(alphas,np.reshape(-cl[i,:,2],(10,)),color='k',marker='o')
    plt.plot(alphas,np.reshape(-cl[i,:,3],(10,)),color='k',marker='s')
    plt.plot(alphas,np.reshape(-cl[i,:,4],(10,)),color='k',marker='^')
    plt.xlim([-6.,16.])
    plt.ylim([-0.01,0.04])
    plt.xlabel(r'$\alpha$ (deg)')
    plt.ylabel(r'$c_l$')
    plt.legend(['-10 deg','-5 deg','0 deg','5 deg','10 deg'])
    plt.title('M='+str(M))
    plt.grid()
    plt.savefig('alpha_v_cl_M'+str(M)+'.png')
    plt.show()
    
    pdb.set_trace()