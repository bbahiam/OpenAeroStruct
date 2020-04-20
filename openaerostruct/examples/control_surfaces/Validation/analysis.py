import numpy as np
import matplotlib.pyplot as plt
from plotting import meshPlot, twistPlot

dels = np.array([-10.,-5.,0.,5.,10.])
qvec = np.array([0.1,10,15,20,25,30,35,40,45,50,55]) * 47.8803
alpha = 0.012 # From Kirsten Wind Tunnel page
rho = 1.2
vels = np.sqrt(2*qvec/rho)
q_eng = qvec/47.8803

Q,D = np.meshgrid(q_eng,dels)

###############################################################################
#                         dcl_dda vs. q straight wing                         #
###############################################################################

data02 = np.array([[9.807,0.0018592],[19.825,0.0015529],[24.87,0.0013926],\
                   [29.776,0.0012038],[34.978,0.000894]])
data04 = np.array([[9.784,0.0042741],
                    [19.961,0.0037898],
                    [24.741,0.0034442],
                    [29.818,0.0029634],
                    [34.823,0.0024825],
                    [39.844,0.0018449]])
data06 = np.array([[9.698,0.0066106],
                    [14.834,0.006251],
                    [19.907,0.0057987]])
data08 = np.array([[9.86,0.0078574],
                   [14.864,0.0073979]])
data95 = np.array([[9.797,0.0085128],
                   [14.808,0.0079749]])

surfname = 'str'
ails =  ['ail02','ail04','ail06','ail08','ail95']
markers=['o','^','s','d','v','P']
colors = ['k','r','b','g','m','c']

plt.figure(1,figsize=(12,7),dpi=300)

plt.scatter(data02[:,0],data02[:,1],marker='x',color=colors[0])
plt.scatter(data04[:,0],data04[:,1],marker='x',color=colors[1])
plt.scatter(data06[:,0],data06[:,1],marker='x',color=colors[2])
plt.scatter(data08[:,0],data08[:,1],marker='x',color=colors[3])
plt.scatter(data95[:,0],data95[:,1],marker='x',color=colors[4])

for i,ail in enumerate(ails):
    cl = np.load(surfname+'_'+ail+'_cl.npy')
    cl_del = np.load(surfname+'_'+ail+'_cl_del.npy')
    plt.plot(q_eng,-cl_del,color=colors[i],linestyle='-')#,marker=markers[i])


plt.ylim([0,0.01])
plt.grid()
plt.xlabel('q (psf)')
plt.ylabel(r'$c_l/\delta_{ail}$')
plt.savefig('reversal_str.png',dpi=300)
plt.show()


###############################################################################
#                          dcl_dda vs. q swept wing                           #
###############################################################################

data02 = np.array([[9.84	,0.0005353],
                    [19.899,0.0003198],
                    [25.027,0.000173],
                    [29.892,0.0001305],
                    [34.822,7.50E-05],
                    [39.883,1.95E-05],
                    [44.878,-3.60E-05]])
data04 = np.array([[20.004,0.0010642],
                    [25.006,0.0008063],
                    [29.877,0.000581],
                    [34.94,0.0004537],
                    [39.938,0.0003264]])
data06 = np.array([[19.975,0.0019586],
                    [24.785,0.0016157],
                    [29.983,0.0013123],
                    [34.918,0.0011262]])
    
data08 = np.array([[19.942,0.0029771],
                    [25.015,0.002543],
                    [29.889,0.0022328],
                    [34.957,0.0019162]])
data95 = np.array([[24.87,0.003013],
                    [29.808,0.0027159],
                    [34.877,0.0023993]])

surfname = 'swp'
ails =  ['ail02','ail04','ail06','ail08','ail95']
markers=['o','^','s','d','v','x']
colors = ['k','r','b','g','m','c']

plt.figure(1,figsize=(12,7),dpi=300)

plt.scatter(data02[:,0],data02[:,1],marker='x',color=colors[0])
plt.scatter(data04[:,0],data04[:,1],marker='x',color=colors[1])
plt.scatter(data06[:,0],data06[:,1],marker='x',color=colors[2])
plt.scatter(data08[:,0],data08[:,1],marker='x',color=colors[3])
plt.scatter(data95[:,0],data95[:,1],marker='x',color=colors[4])

for i,ail in enumerate(ails):
    cl = np.load(surfname+'_'+ail+'_cl.npy')
    cl_del = np.load(surfname+'_'+ail+'_cl_del.npy')
    plt.plot(q_eng,-cl_del,color=colors[i],linestyle='-')#,marker=markers[i])


plt.ylim([0,0.01])
plt.grid()
plt.xlabel('q (psf)')
plt.ylabel(r'$c_l/\delta_{ail}$')
plt.savefig('reversal_swp.png',dpi=300)
plt.show()

import pdb
###############################################################################
#                          Deformed Straight Meshes                           #
###############################################################################
meshes = np.load('str_ail02_defmesh.npy')
base_mesh = np.load('basemesh.npy')
base_vec = base_mesh[-1,:,:] - base_mesh[0,:,:]
base_mag = np.linalg.norm(base_vec)
yvec = meshes[1,0,0,:,1]

velNames = ['q10','q25','q45']
constDel_vVel = [meshes[1,0,:,:,:],meshes[4,0,:,:,:],meshes[8,0,:,:,:]]
for i,mesh in enumerate(constDel_vVel):
    meshPlot(mesh,deformed=True,name=velNames[i]+'.png')
    
    twistPlot(mesh,name=velNames[i]+'_twist.png',minmax=[-4,4])
    
    mesh_vec = mesh[-1,:,:] - mesh[0,:,:]
    
    ang_vec = np.zeros(len(mesh_vec))
    for j in range(len(mesh_vec)):
        ang_vec[j] = 90 - (np.arccos(np.dot(np.array([0,0,-1]),mesh_vec[j,:])\
                           /(np.linalg.norm(mesh_vec[j,:]))) * 180./np.pi)

    plt.figure(1,figsize=(6,2))
    plt.plot(yvec,ang_vec,color='k',linewidth=2)
    plt.xlabel('Y (m)')
    plt.ylabel('Twist (deg)')
    plt.ylim([-3.1,3.1])
    plt.savefig(velNames[i]+'_angs.png')
    plt.show()

#### Make pics for a gif
str_meshes = np.load('str_ail02_defmesh.npy')
swp_meshes = np.load('swp_ail02_defmesh.npy')

for i,q in enumerate(q_eng):
    str_mesh = str_meshes[i,0,:,:,:]
    
    twistPlot(str_mesh,minmax=[-5,5],size=((7,4)),\
              name='str_'+str(i),titleStr='q = '+str(q)+' psf')
    
    swp_mesh = swp_meshes[i,0,:,:,:]

    twistPlot(swp_mesh,minmax=[-5,5],size=((7,4)),\
              name='swp_'+str(i),titleStr='q = '+str(q)+' psf',\
              bounds=[[-0.2,1.02],[-0.61,0.61],[-0.15,0.15]])