from IPython import get_ipython
import matplotlib.pyplot as plt
from UtilityFunc import UF
from VarNet import VarNet
from ADPDE import ADPDE
from ContourPlot import ContourPlot
from Domain import Domain1D
import os

import numpy as np
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
shape = np.shape
reshape = np.reshape

uf = UF()

u = 1                                       # velocity
D = 0.1/pi                                  # diff
T = [0, 2]                                  # time interval
domain = Domain1D(interval=[-1, 1])         # 1D domain

# Initial condition:
def IC(x):
    return -sin(pi*x)


# PDE def:
# Setup AD-PDE:
ADPDE_test = ADPDE(domain, diff=D, vel=u,
                   timeDependent=True, tInterval=T, IC=IC)

# ADPDE_test.plotIC()
# ADPDE_test.plotBC(bInd=0)

VarNet_test = VarNet(ADPDE_test, layerWidth=[20],
                     discNum=20, bDiscNum=None, tDiscNum=300)


# Folder to store Checkpoints:
folderpath = 'c:/Users/sonli/OneDrive/Desktop/Image-Denoising/training_data'
uf.clearFolder(folderpath)

# Backup current operator settings
# uf.copyFile('main.py', folderpath)


VarNet_test.train(folderpath, weight=[1.e1, 1.e1, 1.],
                  smpScheme='optimal', saveFreq=10000, adjustWeight=True)

# %% Simulation results

VarNet_1dt.loadModel()
VarNet_1dt.simRes()
