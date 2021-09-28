import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
This is a simple implementation of perceptron algorithm for binary classification.
We have plant classes and their features. But we only select two features and two plants. Out of 
two features we want to predict which class of plant is it between two classes.

The training function contains of calculate_output and update_weights. Training the algorithm here
 means making determining the correct weights so that it can better predict and classify.
 
In simple terms the training works like this:

1. We calculate the output using the weights we have, then we check if there's an error. 
2. If there's an error we then update the weights based on the error. 

We repeat 1-2 again and again until there's no error or until N times.

Then we draw a boundary line using the formula: w_1*x_1 + x_2*w_2 + b*w_0 = 0

which is converted to  x_2 = -(w_1*x_1 + b*w_0)/w_2

'''



# Getting all data from dataset
dataset = pd.read_csv('iris.csv', header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

dataset.drop(index=dataset.index[dataset['class'] == 'Iris-versicolor'], inplace=True)


# Classes and features to use
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


# initialization

binary_dataset # here is the prepared data
mu = 0.1 # learning rate, will be between 0.1 and 0.4.
b = 1 # bias input
w = [0.15, 0.12, 0.1] # weights, w[0] is bias weight
N = 0 # number of iterations. If set to zero, then it will iterate until there's 0 fail.


# This is sigma, calculating the output. Here we multiply features with weights, 
# and then return the result which is either 1 or 0 (values used to classify, this is 
# binary classification).
def calculate_output(x, w):
    activation = b*w[0]
    for i in range(len(x) - 1):
        activation += w[i+1] * x[i]
    return 1.0 if activation >= 0.0 else 0.0



# This is updating the weights
def update_weights(x, error, w):
    # If there's an error in the calculated output
    # we update the bias weight and other weights. If there's no error, the
    # value of the error is zero, so nothing is updated.
    if error:
        w[0] = w[0] + mu * error
        for i in range(len(w)-1):
            w[i+1] = w[i+1] + mu * error * x[i]
    return


# This is the main function used to train the algorithm. Training the algorithm here means
# making the algorithm have the correct weights so that it can better predict and classify.

def training(w, dataset):
    iterations = 0
    if N > 0: # if N is NOT zero, then it'll iterate N times
        for n in range(N):
            for x in dataset.values:
                y = calculate_output(x, w)
                # we compare calculated output with the desried output 
                # (which is taken from the dataset), gives the error value.
                error = x[-1] - y
                update_weights(x, error, w)
                
    else: # if N is set to zero, then it'll iterate until there's no fail.
        fails = 1
        while fails:  # iteration while there's fail
            fails = 0
            for x in dataset.values:
                y = calculate_output(x, w)
                # we compare calculated output with the desried output 
                # (which is taken from the dataset), gives the error value.
                error = x[-1] - y
                if error: # check if there's error
                    fails += 1 # increment nr of fail since there was an error
                update_weights(x, error, w)

            

# printing initial weights
print(w)

# printing weights at the end
training(w, binary_dataset)
print(w)


# This code is to draw to boundary

# This is to have all the x values.
x = np.linspace(np.min(binary_dataset[feature1]),np.max(binary_dataset[feature1]), 10)
# y = w[0] is weight for bias, w[2] is weight for petal length, w[1] is weight for sepal length

# This is y values for all x values made over, this will draw the boundary line.
# Let's call x[0] for x and x[1] for y:
# w[1]*x + w[2]*y + b*w[0] = 0 
# --> w[2]*y = -(x*w[1] + b*w[0])
# --> y = -(x*w[1] + b*w[0])/w[2]

y = -(x*w[1] + b*w[0])/w[2]




# Plotting the line and everything else
fig, ax = plt.subplots()

for name, group in dataset.groupby('class'):
    ax.scatter(
        group[feature1], group[feature2], label=name)

plt.xlabel(feature1)
plt.ylabel(feature2)

ax.plot(x, y, '-r', label='boundary')

plt.legend()
plt.xlim(binary_dataset[feature1].min()-0.2, binary_dataset[feature1].max()+0.2)
plt.ylim(binary_dataset[feature2].min()-0.2, binary_dataset[feature2].max()+0.2)
plt.show()

