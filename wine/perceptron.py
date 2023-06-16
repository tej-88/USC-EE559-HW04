#Tejas Acharya

import numpy as np

class Perceptron():
    def __init__(self, eta, i_max):
        self.eta = eta
        self.I_MAX = i_max

    def fit(self, X, y, is_version_1=True):
        self.i = 0
        self.C = len(np.unique(y))
        self.N, self.D = X.shape
        self.weights = []

        for i in range(self.C):
            self.weights.append(np.ones((1, self.D)))

        data = np.concatenate((X, y), axis=1)

        if (is_version_1):
            self.sgd_1(data)
        else:
            self.sgd_2(data)
        
        return
    
