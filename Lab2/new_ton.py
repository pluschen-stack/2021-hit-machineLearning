import numpy as np

class NewTon(object):

    def __init__(self,w0,X,Y,lambda_penalty=0,epsilon=1e-5) -> None:
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
        self.lambda_penalty = lambda_penalty
        self.epsilon = epsilon
        self.size = len(self.X)
        self.dim = len(self.X[0])
    
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
        return -loss/self.size

    def __hessianMatrix(self):
        """
        计算海森矩阵
        """
        hessian = self.lambda_penalty * np.eye(self.dim)
        for i in range(self.size):
            temp = self.__sigmoid(self.X[i] @ self.w)
            hessian += self.X[i] * np.transpose([self.X[i]]) * temp * (1 - temp)
        return hessian/self.size
    
    def __gradient(self):
        """
        函数功能:计算梯度
        """
        gradient = self.X.T @ (self.Y - self.__sigmoid(self.X @ self.w))
        return (-gradient + self.lambda_penalty*self.w)/self.size
    
    def train(self):
        hessian = self.__hessianMatrix()
        gradient = self.__gradient()
        old_loss = self.loss(self.X,self.Y)
        new_loss = old_loss
        while True:
            old_loss = new_loss
            self.w = self.w - np.linalg.pinv(hessian) @ gradient
            hessian = self.__hessianMatrix()
            print(gradient.T @ gradient)
            gradient = self.__gradient()
            new_loss = self.loss(self.X,self.Y)
            if np.absolute(old_loss-new_loss) < self.epsilon and gradient.T @ gradient < self.epsilon:
                break
        return self.w
