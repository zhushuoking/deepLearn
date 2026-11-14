#激活函数
import numpy as np
def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return np.maximum(0, x)

def softmax_function(x):
    x = np.exp(x)
    sum_x = np.sum(x)
    return x / sum_x



def tanh_function(x):
    return np.tanh(x)

def identity_function(x):
    return x


#损失函数
#均方误差,y为预测值,t为真实值,一般用于回归问题,对异常值敏感容易梯度爆炸
def mean_squared_error(y, t):
    return np.sum((y - t)**2) * 0.5



#交叉熵误差,y为预测值,t为真实值,一般用于分类问题,对异常值不敏感不容易梯度爆炸
def cross_entropy_error(y, t):
    #将y转化为二维
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    #将t转换为顺序编码(对应的类别标签)
    if t.size == y.size:
        t = t.argmax(axis=1)
    return -np.sum(np.log(y[np.arange(t.size), t] + 1e-7)) / t.size






if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    #t = np.array([1.0, 2.0, 3.0])
    print(step_function(x))
    print(sigmoid_function(x))
    print(relu_function(x))
    #print(mean_squared_error(x,t))
