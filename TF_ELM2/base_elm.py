import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
import numpy as np

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
            self._activation = tf.tanh
        elif activationFun == "relu":
            self._activation = tf.nn.relu
        else:
            raise ValueError(
                "an unknown activation function \'%s\' was given." % (activationFun)
            )
        self._x = tf.placeholder(tf.float32, shape=(None, inputNodes), name='x')
        # self._xi = tf.placeholder(tf.float32, shape=(None, inputNodes), name='xi')
        # self._xb = tf.placeholder(tf.float32, shape=(None, inputNodes), name='xb')
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
        # self._activation = activationFun
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def activationFun_derivative(self, x, name):
        """
        Compute the gradient (slope/derivative) of the activation function with
        respect to its input x.

        Args:
        x: scalar or numpy array

        Returns:
        gradient: gradient of the activation function with respect to x
        """
        if(name == "sigmoid"):    
            y = tf.nn.sigmoid(x)
            return y * (1 - y)
        elif(name == "tanh"):
            y = tf.tanh(x)
            return 1 - y**2

    def hiddenOut(self, x, beta):
        wt = self.activationFun_derivative(tf.add(tf.matmul(self._x, self._weight), self._bias), name = "sigmoid")
        N = tf.matmul(wt, self._beta)
        # return self._sess.run(N, feed_dict={self._x: x})
        # return self._sess.run(N, feed_dict={self._x: x, self._beta: beta})
        return N
    
    def train(self, Input, iInput, bInput):
        vel = 1.0                         # velocity magnitude
        diff = 0.1/np.pi                      # diffusivity
        beta = np.zeros((self.hiddenNodes, self.outputNodes))
        a = self.hiddenOut(Input, beta)
        du = tf.gradients(self.hiddenOut(Input, beta), self._beta)

        H = []
        for i, x in enumerate(Input):
            dC_dx = tf.gradients(self.hiddenOut(x)[i, :], self._x)[0]            # first order derivative wrt time and space
            dC_dt = dC_dx[:,1]
            d2C_dx2 = tf.gradients(dC_dx[:,0], self._x)[0]        # second order derivative
            d2C_dx2 = d2C_dx2[:,0]                          # second order derivative wrt space
            res = dC_dt - diff*d2C_dx2 + vel*dC_dx[:,0]
            H.append(res)
        
        for i in range(iInput.shape[0]):
            phi2 = self.N()
            H.append(phi2)
        for i in range(bInput.shape[0]):
            phi3 = self.N()[i, 0]
            H.append(phi3)

        
        self.H = tf.stack(H)
        
        pass


    def pinv(self, A):
        # Moore-Penrose pseudo-inverse
        with tf.name_scope("pinv"):
            s, u, v = tf.svd(A, compute_uv=True)
            s_inv = tf.reciprocal(s)
            s_inv = tf.diag(s_inv)
            left_mul = tf.matmul(v, s_inv)
            u_t = tf.transpose(u)
            return tf.matmul(left_mul, u_t)
        
    #Calculate regularized least square solution of matrix A
    def regularized_ls(self, A, _lambda):
        shape = A.shape
        A_t = tf.transpose(A)
        if shape[0] < shape[1]:
            _A = tf.matmul(A_t, tf.matrix_inverse(_lambda * np.eye(shape[0]) + tf.matmul(A, A_t)))
        else:
            _A = tf.matmul(tf.matrix_inverse(_lambda * np.eye(shape[1]) + tf.matmul(A_t, A)), A_t)
        return _A


    # def train(self, x, y, name="elm_train"):
        
    #     with tf.name_scope("{}_{}".format(name, 'hidden')):
    #         with tf.name_scope("H"):
    #             h_matrix = tf.matmul(x, self._weight) + self._bias
    #             h_act = self._activation(h_matrix)

    #         h_pinv = self.pinv(h_act)

    #         with tf.name_scope("Beta"):
    #             beta = tf.matmul(h_pinv, y)
    #         return beta


    def inference(self, x, beta, name="elm_inference"):
        with tf.name_scope("{}_{}".format(name, 'out')):
            with tf.name_scope("H"):
                h_matrix = tf.matmul(x, self._weight) + self._bias
                h_act = self._activation(h_matrix)

            out = tf.matmul(h_act, beta)
            return out
        
        
    def compile(self):
        vel = 1.0                         # velocity magnitude
        diff = 0.1/np.pi                      # diffusivity

        phi1 = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))

        H = []
        for i in range(self.hiddenNodes):
            dC_dx = tf.gradients(phi1[:, i], self._x)[0]            # first order derivative wrt time and space
            dC_dt = dC_dx[:,1]
            d2C_dx2 = tf.gradients(dC_dx[:,0], self._x)[0]        # second order derivative
            d2C_dx2 = d2C_dx2[:,0]                          # second order derivative wrt space
            res = dC_dt - diff*d2C_dx2 + vel*dC_dx[:,0]
            H.append(res)
        
        res_result = tf.transpose(tf.stack(H))
        phi2 = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        phi3 = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        
        self.H = tf.concat([res_result, phi2, phi3], axis=0)
        # h_inv = self.pinv(self.H)
        h_inv = self.regularized_ls(self.H, 1 / (self.outputNodes * self.hiddenNodes))

        self._beta = tf.matmul(h_inv, self._y)
        
    def fit1D(self, x_input, y_input):
        # self._sess.run(self.H, feed_dict={self._x: x_input, self._xi: iInput, self._xb: bInput})
        beta = self._sess.run(self._beta, feed_dict={self._x: x_input, self._y: y_input})
        return beta
    
    def evaluate(self, beta, x_test, y_test, itest, btest):
        y_out = tf.matmul(self.H, self._beta)
        # correct_pred = tf.equal(tf.argmax(self._y, 1), tf.argmax(y_out, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # ret = self._sess.run(accuracy, feed_dict={self._x: x_test, self._xi: itest, self._xb: btest, self._y: y_test})
        # a = list(map(lambda x: float(x), ret))

        return self._sess.run(y_out, feed_dict={self._beta: beta})
    
        
        
    def test_1D(self, diff, vel, x_input, y_input, iInput, bInput):
        start = time.perf_counter()
        phi1 = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        
        H = []
        for i in range(self.hiddenNodes):
            dC_dx = tf.gradients(phi1[:, i], self._x)[0]            # first order derivative wrt time and space
            dC_dt = dC_dx[:,1]
            d2C_dx2 = tf.gradients(dC_dx[:,0], self._x)[0]        # second order derivative
            d2C_dx2 = d2C_dx2[:,0]                          # second order derivative wrt space
            res = dC_dt - diff*d2C_dx2 + vel*dC_dx[:,0]
            # x_reshape = x_input[i].reshape((1, x_input[i].size))
            gradients = self._sess.run(res, feed_dict={self._x: x_input})
            H.append(gradients)
        
        res_matrix = np.array(H).T
        
        phi2 = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        phi3 = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        
        ICs = self._sess.run(phi2, feed_dict={self._x: iInput})
        BCs = self._sess.run(phi3, feed_dict={self._x: bInput})
        
        H_matrix = tf.concat([res_matrix, ICs, BCs], axis=0)
        
        # h_inv = self.pinv(H_matrix)
        h_inv = self.regularized_ls(H_matrix, 1 / (self.outputNodes * self.hiddenNodes))

        self._beta = tf.matmul(h_inv, self._y)
        
        result = self._sess.run(self._beta, feed_dict={self._y: y_input})
        end = time.perf_counter()
        print("Training time:", str(end - start))
        
        self.H = H_matrix
        return H_matrix, result
    
    def predict_1D(self, h_matrix, beta):
        y_out = tf.matmul(h_matrix, self._beta)
        return self._sess.run(y_out, feed_dict={self._beta: beta})
    
    
    def predict2(self, x, beta):
        wt = self._activation(tf.add(tf.matmul(self._x, self._weight), self._bias))
        y_out = tf.matmul(wt, self._beta)
        return self._sess.run(y_out, feed_dict={self._x: x, self._beta: beta})
        


    
    def compute_loss(self, logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy
        )

        tf.summary.scalar("loss", cross_entropy_mean)

        return cross_entropy_mean


    def compute_accuracy(self, logits, labels):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy
    