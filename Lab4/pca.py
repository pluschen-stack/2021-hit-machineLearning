import numpy as np

class PCA(object):

    def __init__(self,data,targetDim) -> None:
        """
        data:数据
        targetDim:目标维数
        """
        self.data = self.__deCenter(data)
        self.size = self.data.shape[0]
        self.colums = self.data.shape[1]
        self.targetDim = targetDim
        self.__calVar()
    
    def __deCenter(self,data):
        """
        去中心化
        """
        self.mu = np.mean(data,axis=0)
        return data - self.mu

    def __calVar(self):
        """
        计算协方差矩阵
        """
        self.cov = np.dot(self.data.T,self.data)/self.size


    def train(self):
        """
        计算PCA
        """
        eigenvalues, featureVectors = np.linalg.eig(self.cov)  # 特征值分解
        sortedEigenvalues = np.argsort(eigenvalues)
        # 选取最大的特征值对应的特征向量
        featureVectors = np.delete(featureVectors, sortedEigenvalues[:self.colums - self.targetDim], axis=1)
        return featureVectors, self.mu

    def transform(self,featureVectors):
        """
        将数据从x转化为x帽
        """
        return self.data.dot(featureVectors).dot(featureVectors.T)+self.mu