from __future__ import print_function, division
import numpy as np
from scipy.spatial.transform import Rotation as R

from openmdao.api import ExplicitComponent

class ControlSurface(ExplicitComponent):
    #### TODO
    # - Support arbitrary asymmetric (not antisymmetric) surfaces
    # - Clean up derivative computation to save computation time

    """
    Changes the panel surface normal vectors to account for control
    surface rotation.

    Parameters
    ----------
    delta_aileron : float
        Control surface deflection angle in degrees.
    undeflected_normals : numpy array
        Normal vectors for the panels before control surface deflection.

    Returns
    -------
    deflected_normals: numpy array
        Normal vectors for the panels after control surface deflection.
    """

    def initialize(self):
        self.options.declare('surface', types=dict)  # Is this needed?
        self.options.declare('yLoc',types=list) # Index of Ypos start of aileron, 0=outboard, ny=centerline
        self.options.declare('cLoc',types=list) # Chordwise positions as a fraction of the chord
        self.options.declare('antisymmetric',types=bool) # Antisymmetry (like ailerons)

    def setup(self):
        self.surface = surface = self.options['surface']
        assert len(self.options['yLoc'])==2, "yLoc must contain indexes of begin and end of control surface"
        self.yLoc = self.options['yLoc']
        assert len(self.options['cLoc'])==2, "cLoc must contain chord fractions of begin and end of control surface"
        self.cLoc = self.options['cLoc']
        self.antisymmetric = self.options['antisymmetric']

        mesh = self.surface['mesh']
        nx = self.nx = mesh.shape[0]
        ny = self.ny = mesh.shape[1]

        self.add_input('delta_aileron', val=0, units='deg')
        self.add_input('undeflected_normals', val=np.zeros((nx-1, ny-1, 3)))
        self.add_input('def_mesh', val=np.zeros((nx, ny, 3)), units='m')
        self.add_output('deflected_normals', val=np.zeros((nx-1, ny-1, 3)))

        self.declare_partials('deflected_normals',
                              'undeflected_normals')

        self.declare_partials('deflected_normals',
                              'delta_aileron')


    def compute(self, inputs, outputs):
        deflection = inputs['delta_aileron']*np.pi/180
        normals = inputs['undeflected_normals']
        new_normals = normals.copy()

        surface = self.surface
        yLoc = self.yLoc
        cLoc = self.cLoc
        antisymmetric = self.antisymmetric

        mesh = inputs['def_mesh']

        if ('corrector' in surface['control_surfaces'][0] 
                and surface['control_surfaces'][0]['corrector']):
            deflection *= correction_plain_flap(deflection, cLoc)

############################### Get hinge lines ###############################
        if antisymmetric:
            _, _, mirror_hinge = find_hinge(cLoc, [-y for y in yLoc], mesh)

        h0, h1, hinge = find_hinge(cLoc, yLoc, mesh)


###################### Find affected mesh points ##########################
        self.cs_mesh = cs_mesh = mesh[:,np.min(yLoc):np.max(yLoc)+1,:]

        # X and Y locations of the hingeline
        yHinge = cs_mesh[0,:,1]
        xHinge = (yHinge - h0[1]) * ((h1[0]-h0[0])/(h1[1]-h0[1])) + h0[0]

        # Logical matrix, 1 if point is downstream the hingeline
        xes = cs_mesh[:,:,0]-xHinge
        xes[xes<=0] = 0
        xes[xes>0] = 1
        cs_panels = np.zeros((np.size(xes,axis=0)-1,np.size(xes,axis=1)-1))

        # Normals and areas of the CS mesh
        cs_normals = np.cross(
                        cs_mesh[:-1,  1:, :] - cs_mesh[1:, :-1, :],
                        cs_mesh[:-1, :-1, :] - cs_mesh[1:,  1:, :],
                        axis=2)
        cs_A = 0.5*np.sqrt(np.sum(cs_normals**2, axis=2))

#################### Find deflection interpolation ########################
        # Each panel has 4 points
        # If all 4 are downstream, normal is rotated by commanded deflection
        # If 0<points<=3 are downstream, normal rotated by area-ratio * deflection
        # If 0 are downstream, panel not affected
        for i in range(0,np.size(xes,axis=0)-1):
            for j in range(0,np.size(xes,axis=1)-1):
                mat = np.array([xes[i,j:j+2],xes[i+1,j:j+2]]) # get the 4 points

                if np.sum(mat)==4: # 4 are downstream the hingeline
                    cs_panels[i,j] = 1 # Rotation factor of 1

                elif np.sum(mat)==1: # 1 is downstream the hingeline
                    # get the index of the downstream point
                    ind = np.where(mat==1)
                    ind = np.array([ind[0][0],ind[1][0]])

                    # make 4x4 mesh of relevant gridpoints
                    mat2 = np.array([cs_mesh[i,j:j+2,:],cs_mesh[i+1,j:j+2,:]])
                    mat2[ind[0],ind[1],0] = xHinge[j+ind[1]] # replace downstream point

                    # Get cross product for the area upstream the hingeline
                    crossProd = np.cross(mat2[1,1,:]-mat2[0,0,:],mat2[0,1,:]-mat2[1,0,:])
                    upstream_A = 0.5*np.sqrt(np.sum(crossProd**2))

                    # Put final area ratio
                    cs_panels[i,j] = (cs_A[i,j]-upstream_A)/cs_A[i,j]

                elif np.sum(mat)==2: # 2 are downstream the hingeline
                    ind = np.where(mat==0)
                    ind = np.array([ind[0],ind[1]])

                    mat2 = np.array([cs_mesh[i,j:j+2,:],cs_mesh[i+1,j:j+2,:]])
                    mat2[ind[0,0],ind[1,0],0] = xHinge[j+ind[1,0]]
                    mat2[ind[0,1],ind[1,1],0] = xHinge[j+ind[1,1]]

                    # Get cross product for the area upstream the hingeline
                    crossProd = np.cross(mat2[1,1,:]-mat2[0,0,:],mat2[0,1,:]-mat2[1,0,:])
                    downstream_A = 0.5*np.sqrt(np.sum(crossProd**2))

                    # Put final area ratio
                    cs_panels[i,j] = downstream_A/cs_A[i,j]

                elif np.sum(mat)==3: # 3 are downstream the hingeline
                    # get index of upstream point
                    ind = np.where(mat==0)
                    ind = np.array([ind[0][0],ind[1][0]])

                    # make 4x4 mesh of relevant gridpoints
                    mat2 = np.array([cs_mesh[i,j:j+2,:],cs_mesh[i+1,j:j+2,:]])
                    mat2[ind[0],ind[1],0] = xHinge[j+ind[1]] # replace upstream point

                    # Get cross product for area downstream the hingeline
                    crossProd = np.cross(mat2[1,1,:]-mat2[0,0,:],mat2[0,1,:]-mat2[1,0,:])
                    downstream_A = 0.5*np.sqrt(np.sum(crossProd**2))

                    # Put final area ratio
                    cs_panels[i,j] = downstream_A/cs_A[i,j]
                else:
                    cs_panels[i,j] = 0

        # Cache for partial derivs
        self.cs_panels = cs_panels
        self.hinge = hinge
        self.rot_ang = deflection

        if antisymmetric:
            self.mirror_hinge = mirror_hinge

        # Calculated rotated normals
        for i in range(np.size(cs_panels,axis=0)):
            for j in range(np.size(cs_panels,axis=1)):
                if cs_panels[i,j] != 0:
                    k = j+np.min(yLoc) # y index for normals

                    interp_defl = deflection*cs_panels[i,j]
                    rot = R.from_rotvec(hinge*interp_defl)
                    new_normals[i,k,:] = rot.apply(normals[i,k,:])

                    if antisymmetric:
                        rot = R.from_rotvec(mirror_hinge*interp_defl)
                        new_normals[i,-k-1,:] = rot.apply(normals[i,-k-1,:])

        outputs['deflected_normals'] = new_normals


    def compute_partials(self, inputs, partials):

       # Get cached values
       yLoc = self.yLoc
       cs_panels = self.cs_panels
       hinge = self.hinge
       deflection = self.rot_ang
       antisymmetric = self.antisymmetric
       mirror_hinge = self.mirror_hinge

       normals = inputs['undeflected_normals']
       # Use complex step to get derivatives
       # Not the best but it'll do for now
       # Right now just copy/pasted most from above
       # Could cut down calculations a lot since
       # these are sparse jacobians

       # Get derivative of new_norms wrt normals
       h = 1e-50
       flatNorms = normals.flatten()
       flatNorms = flatNorms*np.ones_like(flatNorms,dtype=complex)

       for q in range(len(flatNorms)):
           flatNorms[q] += complex(0,h)
           flatNorms = flatNorms.reshape(np.shape(normals))
           new_normals = flatNorms*np.ones_like(normals,dtype=complex)

           for i in range(np.size(cs_panels,axis=0)):
               for j in range(np.size(cs_panels,axis=1)):
                   if cs_panels[i,j] != 0:
                       k = j+np.min(yLoc) # y index for normals

                       # Get angle magnitude and axis
                       rot_ang = deflection*cs_panels[i,j]
                       rot_axis = hinge

                       # Convert to rotation matrix
                       rot_K = np.array([[0,-rot_axis[2],rot_axis[1]],
                                         [rot_axis[2],0,-rot_axis[0]],
                                         [-rot_axis[1],rot_axis[0],0]])

                       rot_mat = np.eye(3) + np.sin(rot_ang)*rot_K \
                                   + (1-np.cos(rot_ang))*np.dot(rot_K,rot_K)

                       new_normals[i,k,:] = np.dot(rot_mat, flatNorms[i,k,:])

                       if antisymmetric:
                           # Get angle magnitude and axis
                           rot_ang = deflection*cs_panels[i,j]
                           rot_axis = mirror_hinge

                           # Convert to rotation matrix
                           rot_K = np.array([[0,-rot_axis[2],rot_axis[1]],
                                             [rot_axis[2],0,-rot_axis[0]],
                                             [-rot_axis[1],rot_axis[0],0]])

                           rot_mat = np.eye(3) + np.sin(rot_ang)*rot_K \
                                       + (1-np.cos(rot_ang))*np.dot(rot_K,rot_K)

                           new_normals[i,-k-1,:] = np.dot(rot_mat, flatNorms[i,-k-1,:])

           deriv = np.imag(new_normals/h)
           deriv = deriv.flatten()
           partials['deflected_normals','undeflected_normals'][q,:] = deriv

           flatNorms = flatNorms.flatten()
           flatNorms[q] -= complex(0,h)


       # Get derivative of new_norms wrt deflections
       new_normals = inputs['undeflected_normals']
       new_normals = new_normals*np.ones_like(normals,dtype=complex)
       deflectionCom = deflection * np.ones_like(deflection,dtype=complex)
       deflectionCom += complex(0,h)

       for i in range(np.size(cs_panels,axis=0)):
           for j in range(np.size(cs_panels,axis=1)):
               if cs_panels[i,j] != 0:
                   k = j+np.min(yLoc) # y index for normals

                   interp_defl = deflectionCom*cs_panels[i,j]

                   # Define the rotation vector
                   rot_vec = hinge*interp_defl

                   # Get angle magnitude and axis
                   rot_ang = np.linalg.norm(rot_vec)
                   rot_axis = rot_vec/rot_ang

                   # Convert to rotation matrix
                   rot_K = np.array([[0,-rot_axis[2],rot_axis[1]],
                                     [rot_axis[2],0,-rot_axis[0]],
                                     [-rot_axis[1],rot_axis[0],0]])

                   rot_mat = np.eye(3) + np.sin(rot_ang)*rot_K \
                               + (1-np.cos(rot_ang))*np.dot(rot_K,rot_K)

                   new_normals[i,k,:] = np.dot(rot_mat, normals[i,k,:])

                   if antisymmetric:
                       # Define the rotation vector
                       rot_vec = mirror_hinge*-interp_defl

                       # Get angle magnitude and axis
                       rot_ang = np.linalg.norm(rot_vec)
                       rot_axis = rot_vec/rot_ang

                       # Convert to rotation matrix
                       rot_K = np.array([[0,-rot_axis[2],rot_axis[1]],
                                         [rot_axis[2],0,-rot_axis[0]],
                                         [-rot_axis[1],rot_axis[0],0]])

                       rot_mat = np.eye(3) + np.sin(rot_ang)*rot_K \
                                       + (1-np.cos(rot_ang))*np.dot(rot_K,rot_K)

                       new_normals[i,-k-1,:] = np.dot(rot_mat, normals[i,-k-1,:])

       partials['deflected_normals','delta_aileron'] = np.imag(new_normals/h)
       deflectionCom -= complex(0,h)


def correction_plain_flap(deflection, cLoc):
    # Empirical fudge factor for plain flap effectiveness as a
    # function of the chord percentage and deflection angle
    x = abs(deflection)
    y = 1 - np.sum(cLoc)/2

    if x < 25:
        p00 = 0.6275
        p10 = -0.009711
        p01 = 1.557
        p20 = 0.001847
        p11 = 0.07229
        p02 = -2.092
        p30 = -0.0002002
        p21 = -0.002684
        p12 = -0.1701
        p40 = 8.013e-06
        p31 = -0.0003361
        p22 = 0.01196
        p50 = -1.098e-07
        p41 = 9.761e-06
        p32 = -5.638e-05

        nu = (  p00
              + p10*x
              + p01*y
              + p20*x**2
              + p11*x*y
              + p02*y**2
              + p30*x**3
              + p21*x**2*y
              + p12*x*y**2
              + p40*x**4
              + p31*x**3*y
              + p22*x**2*y**2
              + p50*x**5
              + p41*x**4*y
              + p32*x**3*y**2
              )
    else:
        p00 = 1.668
        p10 = -0.09698
        p01 = -1.174
        p20 = 0.002653
        p11 = 0.1169
        p02 = 1.582
        p30 = -3.266e-05
        p21 = -0.002302
        p12 = -0.1282
        p40 = 1.546e-07
        p31 = 1.141e-05
        p22 = 0.001739

        nu = (  p00
              + p10*x
              + p01*y
              + p20*x**2
              + p11*x*y
              + p02*y**2
              + p30*x**3
              + p21*x**2*y
              + p12*x*y**2
              + p40*x**4
              + p31*x**3*y
              + p22*x**2*y**2
              )

    return nu

def find_hinge(cLoc, yLoc, mesh):
    y0 = np.min(yLoc); # The starting y position of the aileron
    y1 = np.max(yLoc); # The ending y position of the aileron

    mesh0 = mesh[:,y0,:] # The chordwise mesh at the 1st y pos
    mesh1 = mesh[:,y1,:] # The chordwise mesh at the 2nd y pos

    c0 = np.linalg.norm(mesh0[0,:]-mesh0[-1,:]) # Chord length at 1st y pos
    c1 = np.linalg.norm(mesh1[0,:]-mesh1[-1,:]) # Chord length at 2nd y pos

    u0 = (mesh0[-1,:]-mesh0[0,:])/c0 # Unit vector from LE to TE for chord1
    u1 = (mesh1[-1,:]-mesh1[0,:])/c1 # Unit vector from LE to TE for chord2

    h1 = (mesh1[0,:] + cLoc[1]*c1*u1) # second hinge point
    h0 = (mesh0[0,:] + cLoc[0]*c0*u0) # first hinge point
    hinge = h1 - h0 # hinge line
    hinge /= np.linalg.norm(hinge)

    return h0, h1, hinge
