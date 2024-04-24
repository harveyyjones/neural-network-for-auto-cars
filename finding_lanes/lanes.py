import numpy as np
import matplotlib.pyplot as plt

# Function to draw the line
def draw(x1, x2):
    ln = plt.plot(x1, x2)

# Sigmoid function for logistic regression
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

# Function to calculate the cross-entropy error
def calculate_error(line_parameters, points, y):
    n = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1 / n) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy

# Gradient descent function to optimize the parameters
def gradient_descent(line_parameters, points, y, alpha):
    n = points.shape[0]
    for i in range(2000):
        p = sigmoid(points * line_parameters)
        gradient = points.T * (p - y) * (alpha / n)
        line_parameters = line_parameters - gradient
        
        # Extracting weights and bias from the line_parameters
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        
        # Determining the points for the line
        x1 = np.array([points[:,0].min(), points[:,0].max()])
        x2 = -b / w2 + (x1 * (-w1 / w2))
    # Drawing the line
    draw(x1, x2)

# Number of points
n_pts = 100

# Generating random data points for two regions
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

# Initializing line parameters
line_parameters = np.matrix([np.zeros(3)]).T

# Generating labels for the data points
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

# Creating a scatter plot of the data points
_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')

# Running gradient descent to find the optimal line parameters
gradient_descent(line_parameters, all_points, y, 0.06)

# Displaying the plot
plt.show()
