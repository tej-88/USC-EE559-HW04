#Tejas Acharya

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Constants
TRAIN_DATA_FILENAME = './wine_train.csv'
TEST_DATA_FILENAME = './wine_test.csv'

#Load the Data
def load_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, 0:13].to_numpy()
    X0 = np.ones((X.shape[0], 1))
    X = np.concatenate((X0, X), axis=1)
    y = data.iloc[:, -1].to_numpy() - 1
    return X, y.reshape((y.shape[0], 1))
