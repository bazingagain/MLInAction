# Created by leon at 10/11/2017
# 原文: http://www.jianshu.com/p/0da9eb3fd06b
# 本代码仅作学习使用
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

#梯度下降算法的原理是损失函数对于每个参数进行求导，并且利用负梯度对参数进行更新。

#神经网络模型: y = xw
def nn(x, w):
    return x*w
#定义损失函数
def cost(y, t):
    return ((t-y)**2).sum()

#定义梯度下降函数
def gradient(w, x, t):
    return 2*x*(nn(x, w) - t)

# 定义∆w的更新函数
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum()

# 定义目标函数,不加噪音
def f(x):
    return x * 2

# 定义高斯噪音分布,使用正态分布表示,均值为0,方差为0.2 , 即 N(0, 0.2)
x = np.random.uniform(0, 1, 20)
noise_variance = 0.2
noise = np.random.randn(x.shape[0]) * noise_variance + 0
t = f(x) + noise  # 最终的目标函数（带噪音）

w = 0 # 初始权重
learning_rate = 0.1 #学习率
nb_of_iterations = 10 # number of gradient descent updates, 梯度下降10次
w_cost = [(w, cost(nn(x, w), t))] # List to store the weight, costs values
for i in range(nb_of_iterations):
  dw = delta_w(w, x, t, learning_rate) # Get the delta w update
  w = w - dw # Update the current weight parameter
  w_cost.append((w, cost(nn(x, w), t))) # Add weight, cost to list

# Print the final w, and cost
for i in range(0, len(w_cost)):
  print('w({}): {:.4f} \t cost: {:.4f}'.format(i, w_cost[i][0], w_cost[i][1]))

ws = np.linspace(0, 4, num=100)  # weight values
print(ws)
cost_ws = np.vectorize(lambda w: cost(nn(x, w) , t))(ws)  # cost for each weight in ws
plt.plot(ws, cost_ws, 'r-')  # 可视化 权重 与 损失函数的关系
plt.xlabel('$w$', fontsize=15)
plt.ylabel('$\\xi$', fontsize=15)
plt.title('cost vs. weight')
plt.grid()
plt.show()

# Plot the first 2 gradient descent updates
print("tehe len is :%d" % len(w_cost))
plt.plot(ws, cost_ws, 'r-')  # Plot the error curve
# Plot the updates
for i in range(0, len(w_cost)-2):
  w1, c1 = w_cost[i]
  w2, c2 = w_cost[i+1]
  plt.plot(w1, c1, 'bo')
  plt.plot([w1, w2],[c1, c2], 'b-')
  plt.text(w1, c1+0.5, '$w({})$'.format(i))
# Show figure
plt.xlabel('$w$', fontsize=15)
plt.ylabel('$\\xi$', fontsize=15)
plt.title('Gradient descent updates plotted on cost function')
plt.grid()
plt.show()


plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plot the fitted line
plt.plot([0, 1], [0*w, 1*w], 'r-', label='fitted line')
plt.xlabel('input x')
plt.ylabel('target t')
plt.ylim([0,2])
plt.title('input vs. target')
plt.grid()
plt.legend(loc=2)
plt.show()



