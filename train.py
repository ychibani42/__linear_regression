import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the CSV file (assuming it has columns "km" and "price")
data = pd.read_csv('data.csv')

X = data["km"].values
Y = data["price"].values
old_Y = Y
old_X = X
m = float(X.size)
X = np.c_[np.ones(X.shape[0]), X]
X = X / X.max()
Y = Y / Y.max()
learning_rate = 0.01


X = X * old_X.max()


#def csv_reader(path):

def h(thetas, x):
    return (thetas[0] + thetas[1] * x)


def gradient_descent(m, X, Y):
    tmpthetas = np.array([0, 0])
    while (1):
        oldthetas = tmpthetas;
        tmpthetas = tmpthetas - learning_rate * (1 / m) * (X.T @ ((X @ tmpthetas) - Y))
        if np.array_equal(tmpthetas, oldthetas):
            break
    return tmpthetas
            

tmpthetas = gradient_descent(m, X, Y);
tmpthetas[0] = tmpthetas[0] * max(old_Y)
tmpthetas[1] = tmpthetas[1] * (max(old_Y) / max(old_Y))

print(tmpthetas)


# --> line = -0.02 * X + 8469 


plt.xlabel("Kilometrage")
plt.ylabel("Price")
plt.scatter(data['km'].values, data['price'].values)
# plt.plot(X, line)
plt.grid()
plt.show()
