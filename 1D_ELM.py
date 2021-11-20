#%% Modules:

from matplotlib import pyplot as plt
import numpy.linalg as la
from scipy.sparse import coo
from VarNet import VarNet

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
sh = np.shape
reshape = np.reshape

#%% VarNet import
from Domain import Domain1D
from ADPDE import ADPDE
from UtilityFunc import UF
uf = UF()

sess = tf.Session()

from TF_ELM2.base_elm import TF_ELM

#%% 1D AD-PDE: closed-form solution

u = 1.0                         # velocity magnitude
D = 0.1/pi                      # diffusivity
T = 2.0                         # maximum time bound

tlim = [0, T]              # time limits
# Domain limits:
interval = np.array([-1.0, 1.0])
lim = reshape(interval, [2,1])

dim = 1

# BCs (values):
c0 = 0.0;       c1 = 0.0

source = 0.0                            # source field


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



#%% PDE definition:

domain = Domain1D()         # 1D domain

# Setup the AD-PDE:
ADPDE_1dt = ADPDE(domain, diff=D, vel=u, timeDependent=True, tInterval=[0,T], IC=IC,
                  cEx=cExact)

VarNet_1dt = VarNet(ADPDE_1dt,
                    layerWidth=[20],
                    discNum=50,
                    bDiscNum=None,
                    tDiscNum=20)


#%% Training Points:
# Discretize time:
tDiscNum = 10
ht = (tlim[1] - tlim[0])/tDiscNum
t_coordinates = np.linspace(tlim[0]+ht, tlim[1], tDiscNum)
t_coordinates = reshape(t_coordinates, [tDiscNum, 1])


# Discretize domain INTERIOR:
discNum = 50
he = (lim[1]-lim[0])/(discNum)
coordinates = np.linspace(lim[0]+he, lim[1]-he, discNum)
coordinates = np.reshape(coordinates, [discNum, 1])

# Discretize domain boundaries:
bIndNum = 2
bdof = np.ones(bIndNum, dtype=int)
bCoordinates = reshape(lim, [bIndNum, 1, dim])


# Dirichlet boundary conditions:
bInput = []
biDof = []
for bInd in range(bIndNum):
    bInpuTmp = uf.pairMats(bCoordinates[bInd], t_coordinates)
    biDof.append(len(bInpuTmp))                             # number of boundary nodes
    bInput.append(bInpuTmp)                                 # append boundary input
bInput = uf.vstack(bInput)
        


# Input to model
iInput = np.concatenate([coordinates, np.zeros([discNum, 1])], axis=1)      # Initial condition
biInput = uf.vstack([bInput, iInput])                                       # Boundary condition
Input = uf.pairMats(coordinates, t_coordinates)




#%% Training Points:

nt = 10                                         # number of temporal training points
ht = T/nt                                       # element size
Time = np.linspace(0+ht, T, nt)
Time = np.reshape(Time, [nt, 1])

ns = 50                                         # number of spatial training points
hs = (lim[1]-lim[0])/ns                                 # element size
coord = np.linspace(lim[0]+hs, lim[1]-hs, ns)                # spatial training points
coord = np.reshape(coord, [ns, 1])

# Input to model:
Coord = np.reshape(coord, [ns, 1])            # reshape input so that model can evaluate it
Coord2 = np.tile(Coord, [nt, 1])              # repeat spatial training points for each time instace
Time2 = np.repeat(Time, repeats=ns, axis=0)   # time instances corresponding to spatial training points
Input = np.concatenate([Coord2, Time2], axis=1)

# Initial condition input:
iInput = np.concatenate([coord, np.zeros([ns, 1])], axis=1)
ci = IC(coord)

# Boundary condition input:
#Time2 = reshape(np.append([0], Time), [nt, 1])
bInput1 = np.concatenate([lim[0]*np.ones([nt, 1]), Time], axis=1)
bInput2 = np.concatenate([lim[1]*np.ones([nt, 1]), Time], axis=1)
bInput = np.concatenate([bInput1, bInput2], axis=0)






def biTrainData(PDE, biInput, biDof, biArg=[]):
    BCs = PDE.BCs
    domain = PDE.domain
    bIndNum = domain.bIndNum
    
    biArg = [{} for i in range(bIndNum)]
    biArg.append({})
    # Dirichlet boundary conditions:
    bLabel = []
    indp = 0
    for bInd in range(bIndNum):
        ind = indp + biDof[bInd]                    # current end index
        
        # Dirichlet BC data:
        beta = BCs[bInd][1]
        g = BCs[bInd][2]
        
        # Arguments:
        bCoord = biInput[indp:ind, :dim]
        tCoord = [ biInput[indp:ind, dim][np.newaxis].T ]
            
        bLab = g(bCoord, *tCoord, **biArg[bInd])    # compute the boudary function
        bLabel.append(bLab/beta)                    # append corresponding label
        indp = ind                                  # update the previous index
    bLabel = uf.vstack(bLabel)
    
    # Initial condition:
    coord = biInput[ind:, :dim]
    iLabel = PDE.IC(coord, **biArg[-1])
    
    return bLabel, iLabel


biArg, inpArg = [[]]*2
bLabel, iLabel = biTrainData(ADPDE_1dt, biInput, biDof, biArg)
biLabel = uf.vstack([bLabel, iLabel])



sourceFun = lambda x, t=0: source*np.ones([np.shape(x)[0], 1])
diffArg, velArg, sourceArg = [{} for i in range(3)]
tC = [ Input[:,-1][np.newaxis].T ]

sourceVal = sourceFun(Input[:,0:1], *tC, **sourceArg)



#%% ELM
x_input = np.concatenate([Input, iInput, bInput], axis=0)
y_input = np.concatenate([sourceVal, iLabel, bLabel], axis=0)


elm = TF_ELM(inputNodes=2, hiddenNodes=100, outputNodes=1, activationFun="tanh")


elm.train(D, u, Input, y_input, iInput, bInput)
# print(beta)



Coord = Input[:, :dim]
tCoord = Input[:, dim:dim+1]


def cExact2(x,t, trunc = 800):
    # Function to compute the analytical solution as a Fourier series expansion.
    # Inputs:
    #       t: column vector of times
    #       x: row vector of locations
    #       trunc: truncation number of Fourier bases
    
    # Adjust the shape of variables:
    p = np.arange(0, trunc+1.0)
    p = reshape(p, [1, 1, trunc+1])
    t_disc_num = len(t)
    t = reshape(t, [t_disc_num, 1, 1])
    x_disc_num = len(x)
    x = reshape(x, [1, x_disc_num, 1])
    
    cT = 16*pi**2*D**3*u*exp(u/D/2*(x-u*t/2))                           # time solution
    
    cX1_n = (-1)**p*2*p*sin(p*pi*x)*exp(-D*p**2*pi**2*t)                # numerator of first space solution
    cX1_d = u**4 + 8*(u*pi*D)**2*(p**2+1) + 16*(pi*D)**4*(p**2-1)**2    # denominator of first space solution
    cX1 = np.sinh(u/D/2)*np.sum(cX1_n/cX1_d, axis=-1, keepdims=True)    # first component of spacial solution
    
    cX2_n = (-1)**p*(2*p+1)*cos((p+0.5)*pi*x)*exp(-D*(2*p+1)**2*pi**2*t/4)
    cX2_d = u**4 + (u*pi*D)**2*(8*p**2+8*p+10) + (pi*D)**4*(4*p**2+4*p-3)**2
    cX2 = np.cosh(u/D/2)*np.sum(cX2_n/cX2_d, axis=-1, keepdims=True)    # second component of spacial solution
    
    return np.squeeze(cT*(cX1+cX2))

# Plot the exact solution:
# time2 = np.arange(0.0+0.1, T, 0.2)
# coord2 = np.arange(lim[0],lim[1]+0.01,0.01)

time2 = np.linspace(0.0+0.1, T, 5)
coord2 = np.linspace(-1, 1, 200)

# Analytical solution (see the reference for sanity check):
cEx2 = cExact2(coord2, time2)
cInit = reshape(IC(coord2), [1, len(coord2)])
cEx2 = np.concatenate([cInit, cEx2], axis=0)
time2 = np.append(0, time2)


Coord2 = np.transpose(np.tile(coord2, reps=[1, len(time2)]))
Time2 = reshape(np.repeat(time2, repeats=len(coord2)), [len(Coord2), 1])
input2 = np.concatenate([Coord2, Time2], axis=1)
#input2 = uf.pairMats(coord2.reshape(coord2.shape[0], 1), time2.reshape(time2.shape[0], 1))




cEx = cExact(Coord, tCoord)
cEx_VarNet = VarNet_1dt.fixData.cEx

# cApp = elm.predict_1D(h_act, beta)
cApp1 = elm.predict(Input)
cApp2 = elm.predict(input2)
cApp2 = reshape(cApp2, [len(time2), len(coord2)])

appErr = []
# plt.figure()
for t in range(len(time2)):
    plt.plot(coord2, cEx2[t,:], 'b')
    plt.xlabel('x')
    plt.ylabel('concentration')
    plt.grid(True)
    plt.plot(coord2, cApp2[t,:], 'r')
    plt.title('t={0:.2f}s'.format(time2[t]))
    plt.show()
    appErr.append(la.norm(cApp2[t,:] - cEx2[t,:])/la.norm(cEx2[t,:]))

plt.figure()
plt.plot(time2, appErr)
plt.xlabel('t')
plt.ylabel('estimation error')
plt.grid(True)
plt.show()

print('average normalized concentration error: {:.3}'.format(np.mean(appErr)))

# for i in range(cApp1.shape[0]):
#     print("cApp: ", cApp1[i], "\tcEx: ", cEx[i])

string = 'Normalized approximation error: %2.5f' % uf.l2Err(cEx, cApp1)
print(string)
string2 = 'Normalized approximation error new: %2.5f' % uf.l2Err(cEx2, cApp2)
print(string2)

# cApp_test = elm.hiddenOut(Input)
# string2 = 'Normalized approximation error 2: %2.5f' % uf.l2Err(cEx, cApp_test)
# print(string2)

elm.reset()

# nd = 300
# time2_test = np.linspace(0, T, 5)
# xs = np.linspace(-1, 1, nd)
# e_varnet = 0
# e_Elm = 0
# for _t in time2_test:
#     y_exc = cExact(xs.reshape(nd, 1), _t)
#     y_elm = elm.elm(xs, _t, beta).reshape(nd, 1)
#     eElm = np.sum(abs(abs(y_elm) - abs(y_exc))) / np.sum(abs(y_exc))
#     e_Elm += eElm
#     # y_var = VarNet_1dt.evaluate(xs.reshape(nd, 1), _t)
#     # e_var = la.norm(y_var - y_exc) / nd
#     # e_var = np.sum(abs(abs(y_var) - abs(y_exc))) / np.sum(abs(y_exc))
#     # e_varnet += e_var
    
# err_Eml = e_Elm/len(time2_test)
# err_var = e_varnet/len(time2_test)

# print("err_Eml: ", err_Eml)
# print("err_var: ", err_var)
