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
    
    def sgd_1(self, data):
        data = np.random.permutation(data)
        X = data[:, :-1]
        y = data[:, -1]
        num_epochs = 0
        halt_condition = False

        while not halt_condition:
            for i in range(self.N):
                x_i, y_i = X[i], y[i]
                self.update_weights(x_i, y_i)
                self.i += 1
            num_epochs += 1
            halt_condition = self.has_halted(X, y)
        return

    def update_weights(self, x, y):
        g = self.get_g(x)
        k = y
        if np.argmax(g) == k:
            for i in range(self.C):
                self.weights[i] = np.concatenate((self.weights[i], self.weights[i][-1, :]))
            return
        else:
            new_weight_k = self.weights[k][-1, :] + (self.eta * x)
            self.weights[k] = np.concatenate((self.weights[k], new_weight_k))

            l = np.argmax(g[g != g[k]])
            new_weight_l = self.weights[l][-1, :] - (self.eta * x)
            self.weights[l] = np.concatenate((self.weights[l], new_weight_l))

            for m in range(self.C):
                if (m != k) and (m != l):
                    new_weight_m = self.weights[m][-1, :]
                    self.weights[m] = np.concatenate((self.weights[m], new_weight_m))
            return
    

    def get_g(self, x):
        g = np.zeros((self.C, ))
        for i in range(self.C):
            g[i] = np.dot(self.weights[i][-1, :].T, x)
        return g
