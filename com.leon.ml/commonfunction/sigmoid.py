# Created by leon at 01/12/2017

import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return 1/(1 + math.pow(math.e, -x))

x = range(-60, 60)
y = [sigmoid(i) for i in x]

plt.plot(x, y)
plt.xlabel('input x')
plt.ylabel('input y')
plt.xlim([-60, 60])
plt.ylim([0, 1])
plt.title('sigmoid')
plt.grid()
plt.legend(loc=2)
plt.show()


print(math.log(1000000, 2))