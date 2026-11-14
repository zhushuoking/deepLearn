import numpy as np
import matplotlib.pyplot as plt
from common.griadent import numerical_diff


#函数定义
def f(x):
    return 0.01*x**2 + 0.1*x

#切线方程函数,返回切线方程函数 y=kx+b
def tangent_line(f, x):
    y = f(x)
    #计算斜率
    k = numerical_diff(f, x)
    print("斜率k:", k)
    #计算截距
    b = y - k*x
    return lambda t: k*t + b


#定义数据画图
x = np.arange(0.0, 20.0, 0.1)#生成0.0到20.0的数组，步长为0.1
y = f(x)
f_line = tangent_line(f, 5)
plt.plot(x, y, label='f(x)')
y_line = f_line(x)
plt.plot(x, y_line, label='tangent line')

plt.show()



