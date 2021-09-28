import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Getting all data from dataset
dataset = pd.read_csv('iris.csv', header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

dataset.drop(index=dataset.index[dataset['class'] == 'Iris-versicolor'], inplace=True)

class1 = 'Iris-virginica'
class2 = 'Iris-setosa'
feature1 = 'petal length'
feature2 = 'petal width'


# We keep only two classes, representing them as 0 and 1, and we keep only two of the columns

dataset.drop('sepal length', axis='columns', inplace=True)
dataset.drop('sepal width', axis='columns', inplace=True)

binary_dataset = dataset.copy()

binary_dataset.loc[dataset['class'] == class1, dataset.columns == 'class'] = 0
binary_dataset.loc[dataset['class'] == class2, dataset.columns == 'class'] = 1

# comments
# in some DOC file.


# initialization

binary_dataset # here is the prepared data
mu = 0.1 # learning rate
b = 1 # bias
w = [3, 0.12, 0.1] # weights, w(0) is bias weight
N = 0 # number of iterations. If set to zero, then it will iterate until 0 fail.


# This is sigma, calculating the output    
def calculate_output(x, w):
    activation = b*w[0]
    for i in range(len(x) - 1):
        activation += w[i+1] * x[i]
    return 1.0 if activation >= 0.0 else 0.0

# This is updating the weights
def update_weights(x, y, w):
    error = x[-1] - y
    w[0] = w[0] + mu * error
    for i in range(len(w)-1):
        w[i+1] = w[i+1] + mu * error * x[i]
    return


# This is the main function used to train
def training(w, N):
    iterations = 0
    if N > 0: # if N is NOT zero, then it'll iterate N times
        for n in range(N):
            for x in binary_dataset.values:
                y = calculate_output(x, w)
                update_weights(x, y, w)
                
    else: # if N is zero, then it'll iterate until fail.
        fails = 1
        while fails:  # iteration until there's fail
            fails = 0
            for x in binary_dataset.values:
                y = calculate_output(x, w)
                if (x[-1] - y):
                    fails += 1
                update_weights(x, y, w)
                iterations += 1
            print(fails)

        


# printing initial weights
print('weights {}'.format(w))

# printing weights at the end
training(w, N)
print('post weights {}'.format(w))


# This code is to draw to boundary
x = np.linspace(np.min(binary_dataset[feature1]),np.max(binary_dataset[feature1]), 10)

# y = w[0] is weight for bias, w[2] is weight for petal length, w[1] is weight for sepal length
# used the method like desrcibed here: https://medium.com/@thomascountz/calculate-the-decision-boundary-of-a-single-perceptron-visualizing-linear-separability-c4d77099ef38#3778
#y = (-(b*w[0] / w[1]) / (b*w[0] / w[2]))*x + (-b*w[0] / w[1])
# Jan purchase correction
y = -(x*w[1] + b*w[0])/w[2]

# Plotting the line and everything else
fig, ax = plt.subplots()

for name, group in dataset.groupby('class'):
    ax.scatter(
        group[feature1], group[feature2], label=name)

plt.xlabel(feature1)
plt.ylabel(feature2)

ax.plot(x, y, '-r', label='boundary')

plt.grid()
plt.legend()
plt.show()
# plt.xlim(binary_dataset[feature1].min()-0.2, binary_dataset[feature1].max()+0.2)
plt.ylim(binary_dataset[feature2].min()-0.2, binary_dataset[feature2].max()+0.2)

from matplotlib.colors import ListedColormap


def predict(X):
    '''
    yield predictions for the classifier for a set of input values
    '''
    return np.array([calculate_output([x[0], x[1], 0], w) for x in X])


def visualize(X, y, title="Unknown Performance", xName="X", yName="Y", resolution=200):
    '''
    visualize the decision boundary by sampling a load of points throught the space and
    using the classifier to test the prediction at each point
    '''
    X_set, y_set = X, y

    # determine boundarys and calculate safe grid resolution steps
    x0min, x0max = X_set[:, 0].min(), X_set[:, 0].max()
    x1min, x1max = X_set[:, 1].min(), X_set[:, 1].max()
    x0step = (x0max - x0min) / resolution
    x1step = (x1max - x1min) / resolution
    # create meshgrid
    X1, X2 = np.meshgrid(np.arange(start=x0min - 1, stop=x0max + 1, step=x0step),
                         np.arange(start=x1min - 1, stop=x1max + 1, step=x1step))
    plt.contourf(X1, X2, predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.25, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.legend()
    plt.show()


visualize(np.array(binary_dataset),np.array(binary_dataset['class']),title="Perceptron Performance", xName="petal length", yName="petal width")