# Created by leon at 10/11/2017

# 逻辑回归（分类问题）模型

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
from matplotlib import cm

# 目标分类t将从两个独立分布中产生，当t=1时，用蓝色表示。当t=0时，用红色表示。输入参数X是一个N*2的矩阵，目标分类t是一个N * 1的向量

# Define and generate the samples
nb_of_samples_per_class = 20  # 每个类的样本的数目
red_mean = [-1,0]  # The mean of the red class
blue_mean = [1,0]  # The mean of the blue class
std_dev = 1.2  # 两个类的标准偏差
# Generate samples from both classes
# randn -- 20行,2列
x_red = np.random.randn(nb_of_samples_per_class, 2) * std_dev + red_mean
x_blue = np.random.randn(nb_of_samples_per_class, 2) * std_dev + blue_mean

# Merge samples in set of input variables x, and corresponding set of output variables t
X = np.vstack((x_red, x_blue))
t = np.vstack((np.zeros((nb_of_samples_per_class,1)), np.ones((nb_of_samples_per_class,1))))

"""

# Plot both classes on the x1, x2 plane
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='class red')
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='class blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.axis([-4, 4, -4, 4])
plt.title('red vs. blue classes in the input space')
plt.show()
"""

# 定义逻辑函数
def logistic(z):
    return 1/(1 + np.exp(-z))
# 定义神经网络函数  y = 1 / (1 + numpy.exp(-x*w))
def nn(x, w):
    return logistic(x.dot(w.T))
# 定义神经网络预测函数, 根据预测的类别返回1或0
def nn_predict(x,w):
    return np.around(nn(x, w))

#定义损失函数
def cost(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

#定义梯度下降函数
def gradient(w, x, t):
    return (nn(x, w) - t).T * x

# define the update function delta w which returns the
#  delta w for each weight in a vector
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t)


# 描绘损失函数与 权重 w之间的关系
nb_of_ws = 100 # compute the cost nb_of_ws times in each dimension
ws1 = np.linspace(-5, 5, num=nb_of_ws) # weight 1
ws2 = np.linspace(-5, 5, num=nb_of_ws) # weight 2
ws_x, ws_y = np.meshgrid(ws1, ws2) # generate grid
cost_ws = np.zeros((nb_of_ws, nb_of_ws)) # initialize cost matrix
# Fill the cost matrix for each combination of weights
for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        cost_ws[i,j] = cost(nn(X, np.asmatrix([ws_x[i,j], ws_y[i,j]])) , t)

"""
# Plot the cost function surface
plt.contourf(ws_x, ws_y, cost_ws, 20, cmap=cm.pink)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\\xi$', fontsize=15)
plt.xlabel('$w_1$', fontsize=15)
plt.ylabel('$w_2$', fontsize=15)
plt.title('Cost function surface')
plt.grid()
plt.show()
"""

# Set the initial weight parameter
w = np.asmatrix([-4, -2])
# Set the learning rate
learning_rate = 0.05

# Start the gradient descent updates and plot the iterations
nb_of_iterations = 10  # Number of gradient descent updates
w_iter = [w]  # List to store the weight values over the iterations
for i in range(nb_of_iterations):
    dw = delta_w(w, X, t, learning_rate)  # Get the delta w update
    w = w-dw  # Update the weights
    w_iter.append(w)  # Store the weights for plotting

"""
# Plot the first weight updates on the error surface
# Plot the error surface
plt.contourf(ws_x, ws_y, cost_ws, 20, alpha=0.9, cmap=cm.pink)
cbar = plt.colorbar()
cbar.ax.set_ylabel('cost')

# Plot the updates
for i in range(1, 4):
    w1 = w_iter[i-1]
    w2 = w_iter[i]
    # Plot the weight-cost value and the line that represents the update
    plt.plot(w1[0,0], w1[0,1], 'bo')  # Plot the weight cost value
    plt.plot([w1[0,0], w2[0,0]], [w1[0,1], w2[0,1]], 'b-')
    plt.text(w1[0,0]-0.2, w1[0,1]+0.4, '$w({})$'.format(i), color='b')
w1 = w_iter[3]
# Plot the last weight
plt.plot(w1[0,0], w1[0,1], 'bo')
plt.text(w1[0,0]-0.2, w1[0,1]+0.4, '$w({})$'.format(4), color='b')
# Show figure
plt.xlabel('$w_1$', fontsize=15)
plt.ylabel('$w_2$', fontsize=15)
plt.title('Gradient descent updates on cost surface')
plt.grid()
plt.show()

"""


# Plot the resulting decision boundary
# Generate a grid over the input space to plot the color of the
#  classification at that grid point
nb_of_xs = 200
xs1 = np.linspace(-4, 4, num=nb_of_xs)
xs2 = np.linspace(-4, 4, num=nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2) # create the grid
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        classification_plane[i,j] = nn_predict(np.asmatrix([xx[i,j], yy[i,j]]) , w)
# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])

# Plot the classification plane with decision boundary and input samples
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='target red')
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='target blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.title('red vs. blue classification boundary')
plt.show()





