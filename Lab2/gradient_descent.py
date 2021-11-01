import numpy as np
from numpy.lib.function_base import gradient

class GradientDescent(object):
    """
    使用梯度下降方法求解逻辑回归
    """

    def __init__(self,w0,X,Y,alpha=0.03,lambda_penalty=0,epsilon=1e-3) -> None:
        """
        w0: 初始参数 是d*1列向量
        X：特征集 是n*d一个矩阵
        Y：标签集 是n*1的列向量
        lambda_penalty：惩罚项系数
        alpha：学习系数
        epsilon：收敛判断条件
        """
        assert len(X) == len(Y)
        self.w = w0.reshape(w0.shape[0],1)
        self.X = X.reshape(X.shape[0],X.shape[1])
        self.Y = Y.reshape(Y.shape[0],1)
        self.alpha = alpha
        self.lambda_penalty = lambda_penalty
        self.epsilon = epsilon
        self.size = self.X.shape[0]

    def __sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def loss(self,X,Y):
        """
        计算损失函数的值
        X：特征集 是n*d一个矩阵
        Y：标签集 是n*1的列向量
        """
        loss = 0
        for i in range(len(X)):
            loss = loss +  Y[i]*( X[i] @ self.w) - np.log( 1+np.exp(X[i] @ self.w) )
        return -loss/self.size #进行归一化操作

    def __gradient(self):
        """
        函数功能计算梯度
        """
        gradient = self.X.T @ (self.Y - self.__sigmoid(self.X @ self.w))
        return -gradient/self.size #进行归一化操作

    def train(self):
        """
        使用梯度下降法求解w
        """
        gradient = self.__gradient()
        new_loss = self.loss(self.X,self.Y)
        old_loss = new_loss
        while True:
            old_loss = new_loss
            self.w = self.w - self.alpha * (self.lambda_penalty / self.size * self.w  +  gradient)
            gradient = self.__gradient()
            print(gradient.T @ gradient)
            new_loss = self.loss(self.X,self.Y)
            if old_loss <= new_loss: #损失增大说明学习率过大，因此减小学习率
                self.alpha = self.alpha/2
                continue
            elif np.absolute(old_loss-new_loss) < self.epsilon  and gradient.T @ gradient < self.epsilon:
                break
        return self.w.reshape(self.w.shape[0],1)
    
    
    


