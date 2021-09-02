from ELM.ELM import ELM
from Domain import PolygonDomain2D
from ADPDE import ADPDE
from ELM.ELMUtility import ELMData

import numpy as np

#%% PDE input data:

T = 1.5                                 # maximum simulation time
q = [1., 0.]                            # velocity magnitude
kappa = 1.e-3                           # diffusivity
c0 = 1.0                                # concentration value on x=0, |y|<a
a = 0.2                                 # bounds on the concentration BC
nt = 151                                # time discretization number for results

#%% Domain definition and contour plots:

vertices = np.array([[0.0, -0.5], [0.0, -0.2], [0.0, 0.2], [0.0, 0.5], 
                     [2.0, 0.5], [2.0, -0.5]])
domain = PolygonDomain2D(vertices)

#%% AD-PDE:

BC = [[], [0.0, 1.0, c0], [], [], [], []]
ADPDE_2d = ADPDE(domain, diff=kappa, vel=q, tInterval=[0,T], BCs=BC, IC=0.0)

data = ELMData(ADPDE_2d, discNum=[80, 40], bDiscNum=40, tDiscNum=75)
