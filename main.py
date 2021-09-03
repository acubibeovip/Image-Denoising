from matplotlib.pyplot import hsv
from ELM.ELM import ELM
from Domain import PolygonDomain2D
from ADPDE import ADPDE
from ELM.ELMUtility import Domain2D

import numpy as np

#%% PDE input data:

T = 1.5                                 # maximum simulation time
q = [1., 0.]                            # velocity magnitude
kappa = 1.e-3                           # diffusivity
c0 = 1.0                                # concentration value on x=0, |y|<a
vertices = np.array([[0.0, -0.5], [0.0, -0.2], [0.0, 0.2], [0.0, 0.5], 
                     [2.0, 0.5], [2.0, -0.5]])

domain = Domain2D(vertices=vertices, tInterval=[0, T])
hs, coord, ht, t_coord = domain.getMesh(discNum=[80, 40], tDiscNum=75)

print("debugger")


# #%% Domain definition and contour plots:

# vertices = np.array([[0.0, -0.5], [0.0, -0.2], [0.0, 0.2], [0.0, 0.5], 
#                      [2.0, 0.5], [2.0, -0.5]])
# domain = PolygonDomain2D(vertices)

# #%% AD-PDE:

# BC = [[], [0.0, 1.0, c0], [], [], [], []]
# ADPDE_2d = ADPDE(domain, diff=kappa, vel=q, tInterval=[0,T], BCs=BC, IC=0.0)

# mesh = domain.getMesh(discNum=[80, 40], bDiscNum=40)
# data = ELMData(ADPDE_2d, discNum=[80, 40], bDiscNum=40, tDiscNum=75)
