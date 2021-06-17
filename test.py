#%% Modules:

import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
shape = np.shape
reshape = np.reshape
sin = np.sin
cos = np.cos
exp = np.exp
pi = np.pi

from scipy import special
erf = special.erf                       # error function

from Domain import PolygonDomain2D
from ContourPlot import ContourPlot
from ADPDE import ADPDE
from VarNet import VarNet
from UtilityFunc import UF
uf = UF()

import matplotlib.pyplot as plt
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

from UtilityFunc import UF
uf = UF()

#%% PDE input data:

#lim = np.array([[0, -0.5], [2, 0.5]])  # spatial domain boundaries
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

contPlt = ContourPlot(domain, tInterval=[0, T])

# contPlt.animPlot(cExFun)

#%% AD-PDE:

BC = [[], [0.0, 1.0, c0], [], [], [], []]
ADPDE_2d = ADPDE(domain, diff=kappa, vel=q, tInterval=[0,T], BCs=BC, IC=0.0)
#ADPDE_2d.plotBC(1)

#%% Architecture and discretization:

VarNet_2d = VarNet(ADPDE_2d, layerWidth = [10, 20], discNum=[80, 40], bDiscNum=40, tDiscNum=75)

#%% Training:

# Folder to Store Checkpoints:
#folderpath = '/Users/Riza/Documents/Python/TF_checkpoints'
folderpath = 'c:/Users/sonli/OneDrive/Desktop/Image-Denoising/training_data_2D'           # Linux

# VarNet_2d.train(folderpath, weight=[5, 1, 1], smpScheme='uniform')

#%% Simulation results:

VarNet_2d.loadModel(folderpath=folderpath)

sess = VarNet_2d.tfData.sess
graph = VarNet_2d.tfData.graph

check_path = os.path.join(folderpath, 'checkpoint')
def checkpoint(modelid):
    """Construct the file paths and write the checkpoint file."""
    modelpath = repr(os.path.join(folderpath, 'best_model-' + str(modelid)))
    string = 'model_checkpoint_path: ' + modelpath
    string += '\nall_model_checkpoint_paths: ' + modelpath + '\n'
    with open(check_path, 'w') as myfile: myfile.write(string)

iterNum = []
for file in os.listdir(folderpath):
    if ".index" in file:
        ind1 = file.rfind("-") + 1
        ind2 = file.rfind(".")
        iNum = int(file[ind1:ind2])
        iterNum.append(iNum)
iterNum.sort(reverse=True)
meta_path = os.path.join(folderpath, 'best_model-' + str(iterNum[-1]) + '.meta')
for iNum in iterNum:
    # Check if all relevant checkpoint data are available:
    filename = 'best_model-' + str(iNum)
    filepath = os.path.join(folderpath, filename + '.index')
    if not os.path.isfile(filepath): continue
    filepath = os.path.join(folderpath, filename + '.data-00000-of-00001')
    if not os.path.isfile(filepath): continue
        
    # Restore the data:
    checkpoint(iNum)        # create the appropriate checkpoint data
    with graph.as_default():
        saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
        tensorName = "Tensor Name \n\n"
        for tensor in tf.get_default_graph().get_operations():
            casepath = os.path.join(folderpath, 'tensorName.txt')
            myfile = open(casepath, 'w+')
            tensorName += tensor.name + "\n"
        myfile.write(tensorName)
        saver.restore(sess, tf.train.latest_checkpoint(folderpath))
    break
# VarNet_2d.simRes()
