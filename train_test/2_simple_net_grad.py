import numpy as np
from common.functions import *
from common.gradient import *

#定义一个简单的神经网络,只有输入层和输出层

class SimpleNet:
    #初始化
    def __init__(self):
        self.W = np.random.randn(2, 3)

    #前向传播
    def forward(self, x):
        a = np.dot(x, self.W)
        y = softmax_function(a)
        return y


    #计算损失函数
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss
