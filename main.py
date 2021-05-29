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
T = 2.0                                 # time interval
domain = Domain1D(interval=[-1, 1])         # 1D domain

# Initial condition:
def IC(x):
    return -sin(pi*x)

def cExact(x, t, trunc = 800):
    """
    Function to compute the analytical solution as a Fourier series expansion.
    Inputs:
        x: column vector of locations
        t: column vector of times
        trunc: truncation number of Fourier bases
    """
    # Initial condition:
    ind0 = t==0
    cInit = IC(x[ind0])
    
    # Series index:
    p = np.arange(0, trunc+1.0)
    p = reshape(p, [1, trunc+1])
    
    c0 = 16*pi**2*D**3*u*exp(u/D/2*(x-u*t/2))                           # constant
    
    c1_n = (-1)**p*2*p*sin(p*pi*x)*exp(-D*p**2*pi**2*t)                 # numerator of first component
    c1_d = u**4 + 8*(u*pi*D)**2*(p**2+1) + 16*(pi*D)**4*(p**2-1)**2     # denominator of first component
    c1 = np.sinh(u/D/2)*np.sum(c1_n/c1_d, axis=-1, keepdims=True)       # first component of the solution
    
    c2_n = (-1)**p*(2*p+1)*cos((p+0.5)*pi*x)*exp(-D*(2*p+1)**2*pi**2*t/4)
    c2_d = u**4 + (u*pi*D)**2*(8*p**2+8*p+10) + (pi*D)**4*(4*p**2+4*p-3)**2
    c2 = np.cosh(u/D/2)*np.sum(c2_n/c2_d, axis=-1, keepdims=True)       # second component of the solution
    
    # Output:
    c = c0*(c1+c2)
    c[ind0] = cInit
    
    return c


# # Exact solution values for D=0.01/pi:
# if D==0.01/pi:
#     xEx1 = np.array([[0.9, 0.94, 0.96, 0.98, 0.99, 0.999, 1.0]]).T
#     tEx1 = np.array([[0.8]])
#     inpEx1 = uf.pairMats(xEx1, tEx1)
#     cEx1 = np.array([[-0.30516, -0.42046, -0.47574, -0.52913, -0.55393, -0.26693, 0.0]]).T
#     plt.plot(xEx1, cEx1)
    
#     xEx2 = np.array([[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.98, 0.99, 0.999, 1.0]]).T
#     tEx2 = np.array([[1.0]])
#     inpEx2 = uf.pairMats(xEx2, tEx2)
#     cEx2 = np.array([[0.93623, 0.98441, 0.93623, 0.79641, 0.57862, 0.30420, 0.18446, 0.06181, 0.03098, 0.00474, 0.0]]).T
#     plt.plot(xEx2, cEx2)
    
#     xEx3 = xEx1
#     tEx3 = np.array([[1.6]])
#     inpEx3 = uf.pairMats(xEx3, tEx3)
#     cEx3 = np.array([[0.78894, 0.85456, 0.88237, 0.90670, 0.91578, 0.43121, 0.0]]).T
#     plt.plot(xEx3, cEx3)
#     plt.xlabel('x')
#     plt.ylabel('cEx')
#     plt.grid(True)
#     plt.legend(['t=0.8', 't=1.0', 't=1.6'])
#     plt.show()
    
#     inpEx = np.vstack([inpEx1, inpEx2, inpEx3])
#     cEx = np.vstack([cEx1, cEx2, cEx3])
    
#     cExact = None       # disable cExact for advective case


# PDE def:
# Setup AD-PDE:
ADPDE_test = ADPDE(domain, diff=D, vel=u,
                   timeDependent=True, tInterval=[0, T], IC=IC, cEx=cExact)

# ADPDE_test.plotIC()
# ADPDE_test.plotBC(bInd=0)

VarNet_test = VarNet(ADPDE_test, layerWidth=[20],
                     discNum=20, bDiscNum=None, tDiscNum=300)


# Folder to store Checkpoints:
folderpath = 'c:/Users/sonli/OneDrive/Desktop/Image-Denoising/training_data'
# uf.clearFolder(folderpath)

# Backup current operator settings
# uf.copyFile('main.py', folderpath)


# VarNet_test.train(folderpath, weight=[1.e1, 1.e1, 1.],
#                   smpScheme='optimal', saveFreq=10000, adjustWeight=True)

# %% Simulation results

VarNet_test.loadModel(folderpath=folderpath)
VarNet_test.simRes()

# Solution error:
if D==0.1/pi:
    print("\n cEx: ==========================================================\n")
    cEx = VarNet_test.fixData.cEx
    print(cEx)
    print("\n cApp: ==========================================================\n")
    cApp = VarNet_test.evaluate()
    print(cApp)
# else:
#     cApp = VarNet_test.evaluate(x=inpEx[:,0:1], t=inpEx[:,1:2])

string = '\n==========================================================\n'
string += 'Simulation results:\n\n'
string += 'Normalized approximation error: %2.5f' % uf.l2Err(cEx, cApp)
print(string)
VarNet_test.trainRes.writeComment(string)
