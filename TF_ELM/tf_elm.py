import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TF_ELM():
    def __init__(self,
                 inputNodes,
                 outputNodes,
                 hiddenNodes,
                 activationFun,
                 loss="mean_squared_error",
                 name = None):
        
        if name == None:
            self.name = "model"
        else:
            self.name = name
        
        # Activation function
        if activationFun == "sigmoid":
            self.activation = tf.nn.sigmoid
        elif activationFun == "tanh":
            self.activation = tf.tanh
        elif activationFun == "linear" or activationFun == None:
            self.activation = tf.identity
        else:
            raise ValueError("an unknown activation function \'%s\' was given." % (activationFun))
        
        # Loss function
        if loss == 'mean_squared_error':
            self.lossfun = tf.losses.mean_squared_error
        elif loss == 'mean_absolute_error':
            self.lossfun = tf.keras.losses.mean_absolute_error
        elif loss == 'categorical_crossentropy':
            self.lossfun = tf.keras.losses.categorical_crossentropy
        elif loss == 'binary_crossentropy':
            self.lossfun = tf.keras.losses.binary_crossentropy
        else:
            raise ValueError("an unknown loss function \'%s\' was given." % loss)
        
        self.x = tf.placeholder(tf.float32, shape=(None, inputNodes), name="H")
        self.y = tf.placeholder(tf.float32, shape=(None, outputNodes), name="Y")
        
        self.weight = tf.get_variable(
            "weight", 
            shape=[inputNodes, hiddenNodes],
            initializer=tf.random_uniform_initializer(-1,1),
            trainable=False
        )
        self.bias = tf.get_variable(
            "bias",
            shape=[hiddenNodes],
            initializer=tf.random_uniform_initializer(-1,1),
            trainable=False
        )
        self.beta = tf.get_variable(
            "beta",
            shape=[hiddenNodes, outputNodes],
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        self.p = tf.get_variable(
            'p',
            shape=[hiddenNodes, hiddenNodes],
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        
        self.session = tf.Session()
        self.hiddenNodes = hiddenNodes
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        
        self._init_train = self.findOutput()
        self.session.run(tf.global_variables_initializer())
        
        
    def init_train(self, x, y):
        a = self.session.run(self._init_train, feed_dict={self.x: x, self.y: y})
        print(a)
        
    def findOutput(self):
        H = self.activation(tf.matmul(self.x, self.weight) + self.bias)
        HT = tf.transpose(H)
        HTH = tf.matmul(HT, H)
        p = tf.assign(self.p, tf.matrix_inverse(HTH))
        pHT = tf.matmul(p, HT)
        pHTt = tf.matmul(pHT, self.y)
        beta = tf.assign(self.beta, pHTt)
        return beta
    
    def buildModel(self):
        dim = 2
        #%% Define the neural network model:

        model = Sequential()
        model.add(layers.Dense(20, activation='sigmoid', input_shape=(dim+1,), kernel_initializer = 'glorot_uniform'))
        model.add(layers.Dense(1))

        model.summary()

        #%% Derivatives:

        Input = tf.placeholder(tf.float32, name='Input')
        Input.set_shape([None, 3])


        dC_dx = tf.gradients(model(Input), Input)[0]                    # first order derivative wrt time and space
        dC_dt = dC_dx[:,dim:dim+1]                              # time dependent
        dC_dx = dC_dx[:,0:dim]

        d2C_dx2 = tf.gradients(dC_dx[:,0], Input)[0][:,0:1]         # second order derivative
        for d in range(1,dim):                                  # second order derivative wrt space
            d2C_dx2 = d2C_dx2 + tf.gradients(dC_dx[:,d], Input)[0][:,d:(d+1)] 
        
        self.model = model
        self.Input = Input
        self.dC_dt = dC_dt
        self.dC_dx = dC_dx
        self.d2C_dx2 = d2C_dx2

        
    def buildEqSys(self):
        model = self.model
        Input = self.Input
        dC_dx = self.dC_dx
        
        # Define nodes
        iLabel = tf.placeholder(tf.float32, name='iLabel')      # initial labels
        iLabel.set_shape([None, 1])
        biInput = tf.placeholder(tf.float32, name='biInput')                    # boundary nodes
        biInput.set_shape([None, 3])
        biLabel = tf.placeholder(tf.float32, name='biLabel')    # boundary labels
        biLabel.set_shape([None, 1])
        bDof = tf.placeholder(tf.int32, name='bDof') 
        biDimVal = tf.placeholder(tf.float32, name='biDimVal')  # dimensional correction for boundary-initial condition# total number of boundary nodes over space-time
        source = tf.placeholder(tf.float32, name='source')      # source term
        source.set_shape([None, 1])
        gcoef = tf.placeholder(tf.float32, name='gcoef')        # coefficient of \nabla c
        gcoef.set_shape([None, 2])
        N = tf.placeholder(tf.float32, name='N')                # FE basis function values at integration points
        N.set_shape([None, 1])
        dNt = tf.placeholder(tf.float32, name='dNt')            # time-derivative of the FE basis functions at integration points
        dNt.set_shape([None, 1])
        intShape = tf.placeholder(tf.int32, name='intShape')    # number of integration points per element
        detJ = tf.placeholder(tf.float32, name='detJ')          # determinant of the Jacobian
        detJvec = tf.placeholder(tf.bool, shape=[], name='detJvec')
        
        # Boundary conditions
        biCs = biDimVal*model(biInput)
        bCs = biCs[:bDof,0:1]                                   # boundary condition error term
        #bCs = tf.reduce_mean(bCs)

        # Initial conditions:
        iCs = biCs[bDof:,0:1]                               # initial condition error term
        #iCs = tf.reduce_mean(iCs)                           # returns nan for empty tensor
        
        # PDE residual:
        int1 = tf.multiply(dC_dx, gcoef)
        int1 = tf.reduce_sum(int1, axis=-1, keepdims=True)
        int1 = int1 - tf.multiply(model(Input), dNt)
        int1 = tf.reshape(int1, intShape)                       # reshape back for each training point
        int1 = tf.reduce_sum(int1, axis=-1, keepdims=True)        # sum over integration points and elements
        int2 = tf.cond(detJvec,
                        lambda: tf.reduce_sum(detJ*int1),        # sum of errors at training points
                        lambda: detJ*tf.reduce_sum(int1) )       # move 'detJ' outside for computational efficiency
        
        # Source term
        source_term = tf.multiply(source, N)
        
        a = tf.multiply(dC_dx, gcoef)
        a = tf.reduce_sum(a, axis=-1, keepdims=True)
        a = a - tf.multiply(model(Input), dNt)
        # a = tf.reshape(a, intShape)
        a = tf.reduce_sum(a, axis=-1, keepdims=True)
        a = detJ*tf.reduce_sum(a, axis=-1, keepdims=True)
        
        
        # Equation System:
        # H = tf.keras.layers.Concatenate(axis=0)([int2, iCs, bCs])
        # Y = tf.keras.layers.Concatenate(axis=0)([source_term, iLabel, biLabel])
        
        # self.H = H
        # self.Y = Y
        # self.test1 = tf.stack([int2, iCs, bCs])
        self.test1 = tf.concat([a, iCs, bCs], 0)
        self.test2 = tf.concat([source_term, iLabel, biLabel], 0)
        
        self.iLabel = iLabel
        self.biInput = biInput
        self.biLabel = biLabel
        self.bDof = bDof
        self.biDimVal = biDimVal
        self.source = source
        self.gcoef = gcoef
        self.N = N
        self.dNt = dNt
        self.intShape = intShape
        self.detJ = detJ
        self.detJvec = detJvec
        
        self.ICloss = iCs
        self.BCloss = bCs
        self.varLoss = int2
        self.sourceTerm = source_term
        
    def buildDicts(self,
                   Input=None,
                   iLabel=None,
                   biInput=None,
                   biLabel=None,
                   bDofsum=None,
                   measure=None,
                   source=None,
                   gcoef=None,
                   N=None,
                   dNt=None,
                   intShap=None,
                   detJ=None,
                   detJvec=None):
        fdict = {
            self.Input: Input,
            self.iLabel: iLabel,
            self.biInput: biInput,
            self.biLabel: biLabel,
            self.bDof: bDofsum,
            self.biDimVal: measure,
            self.source: source,
            self.gcoef: gcoef,
            self.N: N,
            self.dNt: dNt,
            self.intShape: intShap,
            self.detJ: detJ,
            self.detJvec: detJvec
        }
        return fdict
    
    def test(self, feed_dicts):
        
        sess = self.session
        sess.run(tf.global_variables_initializer())
        
        x = sess.run(self.test1, feed_dict=feed_dicts)
        y = sess.run(self.test2, feed_dict=feed_dicts)
        
        return x, y
        # self.x = tf.stack([int2, ICs, BCs])
        # self.y = tf.stack([source, iLabel, biLabel])
        # self.y = tf.concat([source, iLabel, biLabel], 0)
        # self.x = tf.concat([int2, ICs, BCs], 0)
        
        
        
        