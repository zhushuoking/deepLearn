import numpy as np

#数值微分求导函数
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


#数值微分求梯度,传入的x为向量
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        #利用中心差分公式计算偏导数
        grad[i] = (fxh1 - fxh2) / (2 * h)
        #恢复临时变量的值
        x[i] = tmp
    return grad


#传入的x是一个矩阵
def numerical_gradient(f, x):
    #判断维度
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        for i, x_i in enumerate(x):
            grad[i] = _numerical_gradient(f, x_i)
        return grad

