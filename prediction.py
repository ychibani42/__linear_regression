import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
	description="Predict a car's value using linear regression.")
parser.add_argument('kms', help="car's mileage")
parser.add_argument('--graph', '-g', action='store_true', help='show graph')
options = parser.parse_args()

def predict(theta0, theta1, input):
	return theta0 + theta1 * input


def r_squared(X, Y, theta0, theta1):
    y_mean = np.mean(Y)
    sumofresiduals = np.sum((Y - (theta0 + theta1 * X)) ** 2)
    sumofsquares = np.sum((Y - y_mean) ** 2)
    r2 = 1 - (sumofresiduals / sumofsquares)
    return r2

if __name__ == "__main__":
	try:
		thetas = pd.read_csv("thetas.csv")
		theta0 = thetas.at[0, 'theta0']
		theta1 = thetas.at[0, 'theta1']
	except:
		print("Error with the csv for thetas.")
		exit()
	try:
		data = pd.read_csv("data.csv")
		X = data['km'].values
		Y = data['price'].values
		line = theta0 + theta1 * X
		r_square = r_squared(X, Y, theta0, theta1)
		predicted = predict(theta0, theta1, int(options.kms))
		print("The price for a car with {} km is estimated at {}".format(
			options.kms, round(predicted)))
		print("statistical measure of how well the regression line approximates the actual data : {:0.1%}".format(
			r_square))
	except:
			print("Error file.")
			exit(1)
	if options.graph:
            axes = plt.axes()
            axes = plt.grid()
            plt.scatter(data['km'].values, data['price'].values)
            plt.plot(X, line, c='r')
            plt.title('Linear Regression for price/mileage')
            plt.xlabel('Mileage')
            plt.ylabel('Price')
            plt.show()