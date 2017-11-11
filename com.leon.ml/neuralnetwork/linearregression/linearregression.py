# Created by leon at 10/11/2017

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# Define the vector of input samples as x, with 20 values sampled from a uniform distribution
# between 0 and 1
x = np.random.uniform(0, 1, 20)
# 定义目标函数,不加噪音
def f(x):
    return x * 2

# 定义高斯噪音分布,使用正态分布表示,均值为0,方差为0.2 , 即 N(0, 0.2)
noise_variance = 0.2
noise = np.random.randn(x.shape[0]) * noise_variance + 0
t = f(x) + noise  # 最终的目标函数（带噪音）

plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$t$', fontsize=15)
plt.ylim([0,2])
plt.title('inputs (x) vs targets (t)')
plt.grid()
plt.legend(loc=2)
plt.show()
