import numpy as np

class AnalyticalSolution(object):
    """
    使用最小二乘法求解解析解
    """

    def __init__(self, x_matrix, y_tag):
        """
        x_matrix的维度是N*m，N意味着样本数量，m意味着多项式的阶数,X是可逆的
        y_tag的维度是N*1,N意味着N个样本相对应的目标值
        """
        self.x_matrix = x_matrix
        self.y_tag = y_tag


    def normal(self):
        """
        这个函数返回不带惩罚项的解析解w，w的维度是m*1
        """
        k1 = np.dot(self.x_matrix.T,self.x_matrix)
        k2 = np.linalg.pinv(k1)
        k3 = self.x_matrix.T
        k4 = self.y_tag 
        self.w = np.dot(np.dot(k2,k3),k4)
        return self.w.reshape(self.w.shape[0],1)
        
    
    def with_penalty(self,penalty_lambda):
        """
        这个函数返回带惩罚项的解析解w，w的维度是m*1
        penalty_lambda：惩罚系数λ
        """
        self.penalty_lambda = penalty_lambda
        penalty = self.penalty_lambda*np.eye(self.x_matrix.shape[1])
        self.w = np.linalg.pinv((self.x_matrix.T @ self.x_matrix)+penalty) @ self.x_matrix.T @ (self.y_tag) 
        return self.w.reshape(self.w.shape[0],1)

    def normal_loss(self):
        """
        这个函数返回在训练数据集上的损失函数值
        """
        return (self.x_matrix @ self.w - self.y_tag).T @ (self.x_matrix @ self.w - self.y_tag) * 0.5

    def normal_loss_test(self,x_test_matrix,y_test_tag):
        """
        这个函数返回在测试数据集上的损失函数值
        """
        return (x_test_matrix @ self.w - y_test_tag).T @ (x_test_matrix @ self.w - y_test_tag) * 0.5


    def with_penalty_loss(self):
        """
        这个函数返回带惩罚项的损失函数值
        """
        f = self.penalty_lambda * self.w.T@self.w * 0.5
        return (self.x_matrix @ self.w - self.y_tag).T @ (self.x_matrix @ self.w - self.y_tag) * 0.5 + f

    def with_penalty_loss_test(self,x_test_matrix,y_test_tag):
        """
        这个函数返回带惩罚项的在测试数据集上的损失函数值
        """
        f = self.penalty_lambda * self.w.T@self.w * 0.5
        return (x_test_matrix @ self.w - y_test_tag).T @ (x_test_matrix @ self.w - y_test_tag) * 0.5 + f

    def E_rms(self,y_true,y_pred):
        """
        这个函数计算根均方误差
        """
        return np.sqrt(np.mean(np.square(y_true - y_pred)))