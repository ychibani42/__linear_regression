import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def scaling_data(data, scale):
    return data / scale


def csv_reader(path):
    try:
        data = pd.read_csv(path)
    except:
        print("Error while opening the csv file")
        exit()
    X = np.array(data['km'].values)
    old_X = X
    X = scaling_data(X, max(X))
    X = np.c_[np.ones(X.shape[0]), X]
    Y = np.array(data['price'].values)
    old_Y = Y
    Y = scaling_data(Y, max(Y))
    return X, old_X, Y, old_Y


def z_fct(x, y):
    return np.sin(5 * x) * np.cos(5 * y) / 5


def calculate_3d_gradient(x, y):
    return np.cos(5 * x) * np.cos(5 * y) - np.sin(5 * x) * np.sin(5 * y)


def gradient_descent(m, X, Y, tmpthetas, learning_rate, epoch):

    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    xtmp, ytmp = np.meshgrid(x, y)
    Z = z_fct(xtmp, ytmp)
    ax = plt.subplot(projection="3d", zorder=False)

    for _ in range(epoch):
        oldthetas = tmpthetas
        curr = calculate_3d_gradient(tmpthetas[0], tmpthetas[1])
        tmpthetas = tmpthetas - learning_rate * (1 / m) * (X.T @ ((X @ tmpthetas) - Y))
        if np.array_equal(tmpthetas, oldthetas):
            break
        ax.plot_surface(xtmp, ytmp, Z, cmap="viridis")
        ax.scatter(tmpthetas[0], tmpthetas[1], z_fct(tmpthetas[0], tmpthetas[1]), color="magenta", zorder=1)
        plt.pause(0.0001)
        ax.clear()
    return tmpthetas


def calcul_thetas(X, Y, old_X, old_Y, iterations):
    thetas = gradient_descent(float(len(X)), X, Y, np.array([0, 0]), 0.1, iterations)
    thetas[0] = thetas[0] * max(old_Y)
    thetas[1] = thetas[1] * (max(old_Y) / max(old_X))
    return thetas

def throw_thethas_to_file(tmpthetas):
    with open("thetas.csv", "r+") as fd:
        data = str(tmpthetas[0]) + ',' + str(tmpthetas[1]) + '\n'
        newdata = ",theta0,theta1\n0," + data
        fd.write(newdata)

if __name__ == "__main__":
    X, old_X, Y, old_Y = csv_reader("data.csv")
    # iterations = int(input("Enter a number of iterations needed to find the thetas\n"))
    iterations = 100
    if (iterations < 0 or iterations > sys.maxsize):
        print("That's not a valid number")
    thetas = calcul_thetas(X, Y, old_X, old_Y, iterations)
    throw_thethas_to_file(thetas)

