#%% Modules:

import os
from matplotlib import pyplot as plt
from numpy import linalg as LA
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

from TF_ELM.base_elm import TF_ELM

dim = 1

# BCs (values):
c0 = 0.0;       c1 = 0.0

source = 0.0                            # source field

u = 1.0        
D = 0.1/pi     
T = 2.0 


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
                    tDiscNum=10)



n = 2 
T = 2.
lw = -1.
up = 1.
h_k = 0.1/pi 

nx = 50
nt = 10

xs = np.linspace(lw, up, nx)
ts = np.linspace(0, T, nt)  

T_, X = np.meshgrid(ts, xs)
Dinner = []         
Dinit = []         
Dbound = []
for i in range(nx):
    for k in range(nt):
        x = np.array([X[i][k],T_[i][k]])
        if (x[1]==0): 
            Dinit.append(x)
        elif (x[0]==lw or x[0]==up ):
            Dbound.append(x)
        else:  
            Dinner.append(x)
#Lay mau them tren bien cva dieu kien dau
Dinit = np.array(Dinit)
Dbound = np.array(Dbound)
Dinner = np.array(Dinner)


def init(x): 
    x = x[0]
    return -np.sin(np.pi*x)

Y = []
for i, x in enumerate(Dinner):
    ti = 0 #a0*h_k - c0 - b0 
    Y.append(ti)

for i, x in enumerate(Dinit):
    ti = init(x)
    Y.append(ti)
for i, x in enumerate(Dbound):
    ti = 0
    Y.append(ti)
    
y_input = np.array(Y)
y_input = np.reshape(y_input, [y_input.shape[0], 1])

elm = TF_ELM(inputNodes=2, hiddenNodes=100, outputNodes=1, activationFun="tanh")


elm.train_test(D, u, Dinner, y_input, Dinit, Dbound)


nd = 300
#time2 = np.linspace(0, T, 10)
time2 = np.arange(0, T, 0.5)
xs = np.linspace(-1, 1, nd)
e_Var = 0
e_Eml = 0
for _t in time2:
    y_var = VarNet_1dt.evaluate(xs.reshape(nd,1), _t)
    y_exc = cExact(xs.reshape(nd,1), _t)

    y_eml = elm.eml(xs,_t).reshape(nd,1)
    # plt.plot(xs, y_var, label='Varnet')
    # plt.plot(xs, y_exc, label='Exact')
    # plt.plot(xs, y_eml, label='Eml')
    # plt.xlabel("x");
    # plt.ylabel("y")
    # plt.title("t = {0:.2f}".format(_t))
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    eVar = LA.norm(y_var - y_exc)/nd
    e_Var += eVar
    eEml = LA.norm(y_eml - y_exc)/nd
    e_Eml += eEml
    print("Min error Varnet")
    print(np.min(  np.absolute(np.absolute(y_var) - np.absolute(y_exc))))
    print("Min error Eml")
    print(np.min(  np.absolute(np.absolute(y_eml) - np.absolute(y_exc))))
    print("Max error Varnet")
    print(np.max(  np.absolute(np.absolute(y_var) - np.absolute(y_exc))))
    print("Max error Eml")
    print(np.max(  np.absolute(np.absolute(y_eml) - np.absolute(y_exc))))
    print("errEml: {:.8f}".format(eEml))
    print("errVar: {:.8f}".format(eVar))

print("error Varnet")
print(e_Var/len(time2))
print("error Eml")
print(e_Eml/len(time2))

casepath = os.path.join("c:/Users/sonli/OneDrive/Desktop/Solving-PDE-With-NN", 'result.txt')
myfile = open(casepath, 'w+')
# myfile.write("\terror Eml: " + str(e_Eml/len(time2)))

string = 't       x1       y_eml       y_var        y_exc\
       errEml      errVar\n'
string += '----------------------------------------------------------------------\n'
myfile.write(string)

for t in time2:
    for x in xs:
        n_eml = elm.eml([x],t).item(0)
        n_var = VarNet_1dt.evaluate(np.array(x).reshape(1,1),t).item(0)
        n_exc = cExact(np.array([x]).reshape(1,1),t).item(0)
        errEml = abs(n_eml-n_exc)
        errVar = abs(n_var-n_exc)
        myfile.write("{:.2f}   {:.4f}   {:.8f}  {:.8f}   {:.8f}  {:.8f}  {:.8f}\n"
                     .format(t,x,n_eml,n_var,n_exc,errEml,errVar))
      
      

elm.reset()