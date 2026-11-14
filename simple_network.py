import numpy as np
from common.functions import sigmoid_function,identity_function
#一个简单的神经网络 两个输入 第一层三个神经元 第二层两个神经元 输出两个神经元
#实例暂时没有输出意义
#初始化神经网络
def init_network():
    network = {}#网络
    #第一层的权重和偏置
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    #第二层的权重和偏置
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    #第三层的权重和偏置
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


#前向传播
def forward_propagation(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    #第一层的激活值和输出值
    a1 = np.dot(x, w1) + b1#第一层的计算值
    z1 = sigmoid_function(a1)#激活值
    #第二层的激活值和输出值
    a2 = np.dot(z1, w2) + b2#第二层的计算值
    z2 = sigmoid_function(a2)#激活值
    #第三层的激活值和输出值
    a3 = np.dot(z2, w3) + b3#第三层的计算值
    y = identity_function(a3)#输出值
    return y


#测试主流程
network = init_network()

#定义数据
x = np.array([1.0, 0.5])

#前向传播
y = forward_propagation(network, x)
print(y)

