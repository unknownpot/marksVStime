# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
X = pd.read_csv('Linear_X_Train.csv')
Y = pd.read_csv('Linear_Y_Train.csv')

X = X.values
Y = Y.values


# %%
plt.style.use('seaborn')
plt.scatter(X, Y)
plt.show()


# %%
def hypothesis(x, theta):
    y_ = theta[0] + theta[1]*x
    return y_

def gradiant(X, Y, theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        x = X[i]
        y_ = hypothesis(x, theta)
        y = Y[i]
        grad[0] += (y_ - y)
        grad[1] += (y_ - y) * x

    return grad/m

def error(X, Y, theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        x = X[i]
        y = Y[i]
        y_ = hypothesis(x, theta)
        total_error += (y_ - y)**2

    return total_error/m

def gradDescent(X, Y, max_steps = 100, learning_rate = 0.1):
    theta = np.zeros((2,))
    error_list = []

    for i in range(max_steps):
        grad = gradiant(X, Y, theta)
        e = error(X, Y, theta)
        error_list.append(e)

        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]

    return theta, error_list



# %%
theta, error_list = gradDescent(X, Y)


# %%
theta


# %%
y_ = hypothesis(X, theta)
plt.scatter(X, Y)
plt.plot(X, y_, color = 'orange')
plt.show()


# %%
x_test = pd.read_csv('Linear_X_Test.csv').values
y_test = hypothesis(x_test, theta)


# %%
df = pd.DataFrame(data=y_test, columns = ['y'])


# %%
df.to_csv('y_predictions.csv', index=False)


# %%



