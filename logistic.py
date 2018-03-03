import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# scale values from to values between -1 and 1
min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
df = pd.read_csv("data.csv", header=0)
# clean up data
df.columns = ["grade1", "grade2", "label"]
# remove the trailing ;
x = df["label"].map(lambda x: float(x.rstrip(';')))

# formats the input data into two arrays, one of independent variables
# and one of the dependant variable
X = df[["grade1", "grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)

# creating testing and training set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
# parameter set initialized as zero
theta = [0, 0]


# The sigmoid function
def sigmoid(z: float):
    sigmoid_of_z = float(1 / float(1 + math.exp(-z)))
    return sigmoid_of_z


def sigmoid_of_list(r: range):
    return_value = []
    for i in r:
        return_value.append(sigmoid(i))
    return return_value


def sigmoid_with_theta(x: list, theta: list):
    if len(x) != len(theta):
        raise ValueError
    parameter = 0
    for i in range(len(x)):
        parameter += x[i] * theta[i]
    return sigmoid(parameter)


def cost(expected_y: float, real_y: bool):
    if real_y:  # y == 1
        return -math.log2(expected_y)
    else:
        return -math.log2(1 - expected_y)


def cost_of_list(expected_y: list, real_y: list):
    if len(expected_y) != len(real_y):
        raise ValueError  # The lengths should be the same
    length = len(expected_y)
    total = 0
    for i in range(length):
        total += cost(expected_y[i], real_y[i])
    return total / length


def derivative_cost(x, y, theta, j):
    if len(x) != len(y):
        raise ValueError
    total = 0
    length = len(x)
    for i in range(length):
        total += (sigmoid_with_theta(x[i], theta) - y[i])*x[i][j]
    return total / length


def total_difference(x: list, y: list):
    if len(x) != len(y):
        raise ValueError
    total = 0
    length = len(x)
    for i in range(length):
        total = math.fabs(x[i] - y[i])
    return total


# alpha is the learning rate, theta the list of parameters
def gradient_descent(alpha: float, theta_list: list, x, y):
    while True:
        theta_before = list(theta_list)  # copy the old parameters
        for j in range(len(theta_list)):
            theta[j] -= alpha * derivative_cost(x, y, theta_list, j)
        if total_difference(theta_before, theta_list) < 0.0000000000000000001:
            break


gradient_descent(1, theta, X_train, Y_train)
expected_values = [0.1] * len(X_test)  # make expected_values a list with the same length as X_test, make the type floats
for i in range(len(expected_values)):
    expected_values[i] = sigmoid_with_theta(X_test[i], theta)
print(cost_of_list(expected_values, Y_test))
