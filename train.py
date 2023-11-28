import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Load data from the CSV file (assuming it has columns "km" and "price")
data = pd.read_csv('data.csv')


def scaling_data(data, scale):
	"""Scaling des data"""
	return data / scale


def csv_reader(path):
    data = pd.read_csv(path)
    X = np.array(data['km'].values)
    old_X = X
    X = scaling_data(X, max(X))
    X = np.c_[np.ones(X.shape[0]), X]
    Y = np.array(data['price'].values)  # OK
    old_Y = Y
    Y = scaling_data(Y, max(Y))
    return X, old_X, Y, old_Y


def gradient_descent(m, X, Y, tmpthetas, learning_rate, epoch):
    tmpthetas = np.array([0, 0])
    for i in range(epoch):
        oldthetas = tmpthetas
        tmpthetas = tmpthetas - learning_rate * (1 / m) * (X.T @ ((X @ tmpthetas) - Y))
        if np.array_equal(tmpthetas, oldthetas):
            break
    return tmpthetas


def calcul_thetas(X, Y, old_X, old_Y):
    thetas = gradient_descent(float(len(X)), X, Y, np.array([0, 0]), 1, 100000)
    thetas[0] = thetas[0] * max(old_Y)
    thetas[1] = thetas[1] * (max(old_Y) / max(old_X))
    return thetas


def throw_thethas_to_file(thethas):
	"""Enregistrement des thetas dans le csv"""
	with open("thetas.csv", "r+") as fd:
		data = str(thethas[0]) + ',' + str(thethas[1]) + '\n'
		newdata = ",theta0,theta1\n0," + data
		fd.write(newdata)


def accuracy(X, Y, thetas):
    y_mean = Y.mean()
    sumofsquares = 0
    sumofresiduals = 0
    for i in range (len(X)):
         y_pred = thetas[0] + thetas[1] * X[i]
         sumofsquares += (Y[i] - y_mean) ** 2
         sumofresiduals += (Y[i] - y_pred) ** 2
    score = 1 - (sumofsquares / sumofresiduals)
    return (score)

X, old_X, Y, old_Y = csv_reader("data.csv")
thethas = calcul_thetas(X, Y, old_X, old_Y)
throw_thethas_to_file(thethas)
score = accuracy(old_X, old_Y, thethas)
print(score)
