import numpy as np

class GradientDescent(object):

    def __init__(self,x_matrix,y_tag,penalty_lambda,alpha,delta=1e-5):
        """
        x_matrix的维度是N*m，N意味着样本数量，m意味着多项式的阶数
        y_tag的维度是N*1,N意味着N个样本相对应的目标值
        penalty:惩罚系数
        alpha：学习率
        delta：当梯度小于等于delta时停止算法
        """
        self.x_matrix = x_matrix
        self.y_tag = y_tag.reshape(y_tag.shape[0],1)
        self.penalty_lambda = penalty_lambda
        self.alpha = alpha
        self.delta = delta

    def __gradient(self,w):
        """
        计算当前的梯度
        w:当前的解w
        """
        k1 = self.x_matrix.T @ self.x_matrix @ w
        k2 = self.x_matrix.T @ self.y_tag
        k3 = self.penalty_lambda * w
        return k1 - k2 + k3
    
    def __loss(self,w):
        """
        计算当前的损失值
        """
        f = self.penalty_lambda * w.T@w * 0.5
        return (self.x_matrix @ w - self.y_tag).T @ (self.x_matrix @ w - self.y_tag) * 0.5 + f

    def train(self):
        """
        获取模型结果
        返回轮数，结果，每一轮的损失值
        """
        w = np.zeros((self.x_matrix.shape[1], 1))
        k = 1
        current_gradient = self.__gradient(w)
        rounds = []
        loss = []
        while not np.absolute(current_gradient.T @ current_gradient) <= self.delta:
            w = w - self.alpha * current_gradient
            print(current_gradient)
            current_gradient = self.__gradient(w)
            rounds.append(k)
            loss.append(self.__loss(w))
            k+=1
            if np.isnan(w).any() :
                break
        return np.array(rounds),w,np.array(loss)