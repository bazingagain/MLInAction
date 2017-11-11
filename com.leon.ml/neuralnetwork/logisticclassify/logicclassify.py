# Created by leon at 10/11/2017
# 原文:http://www.jianshu.com/p/abc2acf092a3
# 本代码仅作学习使用
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

def logistic(z):
    """
    逻辑函数
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))

def logistic_derivative(z):
    """
    逻辑函数的倒数, 也是其梯度
    :param z:
    :return:
    """
    return logistic(z) * (1 - logistic(z))

# z = np.linspace(-6,6,100)
# plt.plot(z, logistic(z), 'b-')
# plt.xlabel('$z$', fontsize=15)
# plt.ylabel('$\sigma(z)$', fontsize=15)
# plt.title('logistic function')
# plt.grid()
# plt.show()

z = np.linspace(-6,6,100)
plt.plot(z, logistic_derivative(z), 'r-')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$\\frac{\\partial \\sigma(z)}{\\partial z}$', fontsize=15)
plt.title('derivative of the logistic function')
plt.grid()
plt.show()
