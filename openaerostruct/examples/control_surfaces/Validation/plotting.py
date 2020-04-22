import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import pdb

def surfForcePlot(prob):
    #mesh = prob['AS_point_0.coupled.wing.mesh']
    def_mesh = prob['AS_point_0.coupled.wing.def_mesh']
    forces = prob['AS_point_0.coupled.aero_states.wing_mesh_point_forces']
    
    # Get XYZ forces and normalize them
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
      
    import pdb; pdb.set_trace()
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
    
def meshPlot(mesh,azim=225,elev=45,deformed=False,name=None,showIt=True,
             bounds=[[-0.61,0.61],[-0.61,0.61],[-0.15,0.15]],axisEqual=False,\
             size=(12,6)):    
    X = mesh[:,:,0];
    Y = mesh[:,:,1];
    Z = mesh[:,:,2];
    
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), 0.2-(-0.2)]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())    
      
    # Plot X forces
    fig1 = plt.figure(1,figsize=size,dpi=500)
    ax1 = fig1.gca(projection='3d')
    
    if deformed==True: 
        
        surf = ax1.plot_surface(
         X, Y, Z, rstride=1, cstride=1,
        cmap='coolwarm',
        linewidth=0, antialiased=False, shade=False)
        
        m = cm.ScalarMappable(cmap=surf.cmap, norm=surf.norm)
        m.set_array(Z)
        cbar = plt.colorbar(m)
        m.set_clim([-0.025,0.025])
        cbar.set_label('Z deflection (m)')
    else:
        ax1.plot_surface(
            X, Y, Z, rstride=1, cstride=1,
            color='w',
            linewidth=0, antialiased=False, shade=True)
    
    if axisEqual:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax1.plot([xb], [yb], [zb], 'w')
       
    ax1.plot_wireframe(X,Y,Z,color='k',linewidth=0.25)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    if bounds is not None:
        ax1.set_xlim3d(bounds[0])    
        ax1.set_ylim3d(bounds[1])
        ax1.set_zlim3d(bounds[2])
    
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='z', nbins=3)

    ax1.elev=elev
    ax1.azim=azim
    
    #plt.grid()
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    plt.axis('off')
    if name is not None:
        plt.savefig(name+'.png',dpi=900)
        
    if showIt:
        plt.show()
   # import pdb;pdb.set_trace();


def twistPlot(mesh,azim=225,elev=45,name=None,showIt=True,minmax=[-3,3],\
             bounds=[[-0.61,0.61],[-0.61,0.61],[-0.15,0.15]],axisEqual=False,\
             size=(12,6),titleStr=None):    
    X = mesh[:,:,0];
    Y = mesh[:,:,1];
    Z = mesh[:,:,2];
    
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), 0.2-(-0.2)]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())    


    # Plot X forces
    fig1 = plt.figure(1,figsize=size,dpi=300)
    ax1 = fig1.gca(projection='3d')
    
    mesh_vec = mesh[-1,:,:] - mesh[0,:,:]
    
    ang_vec = np.zeros(len(mesh_vec))
    for j in range(len(mesh_vec)):
        ang_vec[j] = 90 - (np.arccos(np.dot(np.array([0,0,-1]),mesh_vec[j,:])\
                           /(np.linalg.norm(mesh_vec[j,:]))) * 180./np.pi)
    
    angs = np.reshape(ang_vec,(1,len(ang_vec))) * np.ones(np.shape(mesh[:,:,0]))
        
    
    # fourth dimention - colormap
    # create colormap according to x-value (can use any 50x50 array)
#    

    norm = matplotlib.colors.Normalize(minmax[0],minmax[1])
    m = cm.ScalarMappable(norm=norm, cmap='coolwarm')
    m.set_array([])
    fcolors = m.to_rgba(angs)
    
    surf = ax1.plot_surface(X, Y, Z, facecolors=fcolors,
         linewidth=0, antialiased=False, shade=False)
    
    cbar = plt.colorbar(m)#,orientation='horizontal')
    cbar.set_label('Twist (deg)')
    
    if axisEqual:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax1.plot([xb], [yb], [zb], 'w')
       
    ax1.plot_wireframe(X,Y,Z,color='k',linewidth=0.25)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    if bounds is not None:
        ax1.set_xlim3d(bounds[0])    
        ax1.set_ylim3d(bounds[1])
        ax1.set_zlim3d(bounds[2])
    
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='z', nbins=3)
    
    if titleStr is not None:
        plt.title(titleStr)
        
    ax1.elev=elev
    ax1.azim=azim
    
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #plt.grid()
    if name is not None:
        plt.savefig(name+'.png',dpi=1200)
        
    if showIt:
        plt.show()
   # import pdb;pdb.set_trace();