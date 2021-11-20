import time
from datetime import timedelta
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
import numpy as np
import torch

class TF_ELM():
    def __init__(self,
                 inputNodes,
                 hiddenNodes,
                 outputNodes,
                 activationFun="sigmoid"):
        
        if activationFun == "sigmoid":
            self._activation = tf.nn.sigmoid
        elif activationFun == "linear" or activationFun == None:
            self._activation = tf.identity
        elif activationFun == "tanh":
            self._activation = tf.nn.tanh
        elif activationFun == "relu":
            self._activation = tf.nn.relu
        else:
            raise ValueError(
                "an unknown activation function \'%s\' was given." % (activationFun)
            )
        self._x = tf.placeholder(tf.float32, shape=(None, inputNodes), name='x')
        self._xi = tf.placeholder(tf.float32, shape=(None, inputNodes), name='xi')
        self._xb = tf.placeholder(tf.float32, shape=(None, inputNodes), name='xb')
        self._y = tf.placeholder(tf.float32, shape=(None, outputNodes), name='y')
        self._beta = tf.placeholder(tf.float32, shape=[hiddenNodes, outputNodes], name="beta_placeholder")
        # self._beta = tf.Variable(
        #     tf.random_uniform(shape=[hiddenNodes, outputNodes]),
        #     dtype='float32', 
        #     trainable=False
        # )
                
        self._weight = tf.get_variable(
            "weight",
            dtype=tf.float32,
            shape=[inputNodes, hiddenNodes],
            initializer=tf.random_normal_initializer()
        )
        self._bias = tf.get_variable(
            "bias",
            dtype=tf.float32,
            shape=[hiddenNodes],
            initializer=tf.random_normal_initializer()
        )
        
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def pinv(self, A):
        # Moore-Penrose pseudo-inverse
        with tf.name_scope("pinv"):
            s, u, v = tf.svd(A, compute_uv=True)
            s_inv = tf.reciprocal(s)
            s_inv = tf.diag(s_inv)
            left_mul = tf.matmul(v, s_inv)
            u_t = tf.transpose(u)
            return tf.matmul(left_mul, u_t)
        
    def p_inv(self, matrix, _lambda=1e-6):
        
        """Returns the Moore-Penrose pseudo-inverse"""
        print("lamda: ", _lambda)
        s, u, v = tf.svd(matrix)
        
        threshold = tf.reduce_max(s) * _lambda
        s_mask = tf.boolean_mask(s, s > threshold)
        s_inv = tf.diag(tf.concat([1. / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))

        return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))
        
    #Calculate regularized least square solution of matrix A
    def regularized_ls(self, A, _lambda):
        shape = A.shape
        A_t = tf.transpose(A)
        if shape[0] < shape[1]:
            _A = tf.matmul(A_t, tf.matrix_inverse(_lambda * np.eye(shape[0]) + tf.matmul(A, A_t)))
        else:
            _A = tf.matmul(tf.matrix_inverse(_lambda * np.eye(shape[1]) + tf.matmul(A_t, A)), A_t)
        return _A
        
    def train(self, diff, vel, x_input, y_input, iInput, bInput):
        start = time.perf_counter()
        phi1 = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        
        H = []
        for i in range(self.hiddenNodes):
            dC_dx = tf.gradients(phi1[:, i], self._x)[0]            # first order derivative wrt time and space
            dC_dt = dC_dx[:,1]
            d2C_dx2 = tf.gradients(dC_dx[:,0], self._x)[0]        # second order derivative
            d2C_dx2 = d2C_dx2[:,0]                          # second order derivative wrt space
            res = dC_dt - diff*d2C_dx2 + vel*dC_dx[:,0]

            gradients = self._sess.run(res, feed_dict={self._x: x_input})
            H.append(gradients)
        
        res_matrix = np.array(H).T
        
        phi2 = self._activation(tf.add(tf.matmul(self._xi, self._weight), self._bias))
        phi3 = self._activation(tf.add(tf.matmul(self._xb, self._weight), self._bias))
        
        ICs = self._sess.run(phi2, feed_dict={self._xi: iInput})
        BCs = self._sess.run(phi3, feed_dict={self._xb: bInput})
        
        H_matrix = tf.concat([res_matrix, ICs, BCs], axis=0)
        
        # h_inv = self.pinv(H_matrix)
        # h_inv = self.regularized_ls(H_matrix, 1 / (self.outputNodes * self.hiddenNodes))
        # h_inv = self.p_inv(H_matrix, 1e-6)
        h_inv = tf.linalg.pinv(H_matrix)

        self._beta = tf.matmul(h_inv, self._y)
        
        self._beta = self._sess.run(self._beta, feed_dict={self._y: y_input})
        end = time.perf_counter()
        print("Hidden nodes:", self.hiddenNodes)
        print("Training time:", str(timedelta(seconds=(end - start))))
    
    
    def predict(self, x):
        wt = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        y_out = tf.matmul(wt, self._beta)
        return self._sess.run(y_out, feed_dict={self._x: x})
     
    # def sigmoid(self, z):
    #     y= np.exp(-z)
    #     y1 = np.exp(z)
    #     return (y1 - y) / (y1 + y)
    
     
    # def N(self, x, beta):
    #     bias = self._sess.run(self._bias)
    #     W = self._sess.run(self._weight)
    #     wt = self.sigmoid(np.add(np.dot(x, W), bias))
    #     return ((np.dot(wt, beta)))
        

    # def elm(self, x, t, beta):
    #     em = np.zeros_like(x)
    #     for i, v in enumerate(x):
    #         em[i] = self.N(np.array([v, t]), beta)
    #     return em

    def reset(self):
        self._sess.close()
    