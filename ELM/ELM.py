import numpy as np
from scipy.linalg import pinv2, inv
import time

class ELM():
    def __init__(self, hiddenNodes, activationFun, x, y, randomType="normal"):         
        if randomType == "uniform":
            self.weight = np.random.uniform(size=(x.shape[1], hiddenNodes))
            self.bias = np.random.uniform(size=(hiddenNodes))
        elif randomType == "normal":
            self.weight = np.random.normal(size=(x.shape[1], hiddenNodes))
            self.bias = np.random.normal(size=(hiddenNodes))
            
        self.hiddenNodes = hiddenNodes
        self.activationFun = activationFun
        self.x = x
        self.y = y
        self.randomType = randomType
        self.class_num = np.unique(y).shape[0]     
        self.beta = np.zeros((self.hiddenNodes, self.class_num))
    
    '''
    Compute the output of hidden layer according to activation function
    '''
    def findHiddenOutput(self, x):
        self.tempH = np.dot(x, self.weight) + self.bias
        
        if self.activationFun == "":
            self.H = self.tempH
        elif self.activationFun == "sigmoid":
            self.H = 1/(1 + np.exp(-self.tempH))
        elif self.activationFun == "sin":
            self.H = np.sin(self.tempH)
        elif self.activationFun == "tanh":
            self.H = (np.exp(self.tempH) - np.exp(-self.tempH))/(np.exp(self.tempH) + np.exp(-self.tempH))
        
        return self.H
    
    def findOutputWeight(self, x):
        self.output = np.dot(x, self.beta)
        return self.output
    
    def fit(self):
        self.time1 = time.perf_counter()
        
        self.H = self.findHiddenOutput(self.x)
        self.beta = np.dot(pinv2(self.H), self.y) # ouput weights
        
        self.time2 = time.perf_counter()
        
        train_time = str(self.time2 - self.time1)
        
        print("Training time:", train_time)
        # return self.beta, train_time
    
    
    '''
    Testing
    '''
    def predict(self, x):
        self.H = self.findHiddenOutput(x)
        self.out = self.findOutputWeight(self.H)
        return self.out
        