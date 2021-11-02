import collections
import numpy as np
from scipy.stats import multivariate_normal

class GMM(object):

    def __init__(self,data,k,delta=1e-3) -> None:
        self.data = data
        self.k = k
        self.delta = delta
        self.dataSize = data.shape[0]
        self.dim = data.shape[1]
        self.initializeParameter()
    
    def initializeParameter(self):
        self.alpha = [1/self.k for i in range(self.k)]
        self.sigma = [np.eye(self.dim) for i in range(self.k)]
        self.initializeMu()
        self.__lastAlpha = np.array(self.alpha) #保存上一次的混合系数
        self.__lastMu = np.array(self.mu) #保存上一次的均值
        self.__lastSigma = np.array(self.sigma) #保存上一次的协方差矩阵
        print('initial:',self.mu)

    def __likelihood(self):
        """
        计算似然值
        """
        total = 0
        for j in range(self.dataSize):
            total += np.sum([self.alpha[i]*multivariate_normal.pdf(self.data[j],self.mu[i],self.sigma[i]) for i in range(self.k)])
        return np.log(total)

    def __eStep(self):
        """
        EM算法E步
        计算样本中的数据由各混合成分生成的后验概率
        使用multivariate_normal.pdf函数计算多元正态分布的值
        """
        self.gamma = np.zeros((self.dataSize,self.k))
        for j in range(self.dataSize):
            total = np.sum([self.alpha[i]*multivariate_normal.pdf(self.data[j],self.mu[i],self.sigma[i]) for i in range(self.k)])
            for i in range(self.k):
                self.gamma[j][i] = self.alpha[i]*multivariate_normal.pdf(self.data[j],self.mu[i],self.sigma[i])\
                    /total

    def __mStep(self):
        """
        EM算法M步
        """
        for i in range(self.k):
            gamma = np.expand_dims(self.gamma[:, i], axis=1)
            self.mu[i] = np.sum(gamma*self.data,axis=0)/gamma.sum()
            self.sigma[i] =  (self.data - self.mu[i]).T.dot((self.data - self.mu[i]) * gamma)/ gamma.sum()
            self.alpha[i] = gamma.sum() / self.dataSize  

    def __converged(self):
        """
        用来判断是否收敛
        """
        difference = self.euclideanDistance(np.array(self.mu),self.__lastMu)\
            +self.euclideanDistance(np.array(self.sigma),self.__lastSigma)\
            +self.euclideanDistance(np.array(self.alpha),self.__lastAlpha)
        if difference > self.delta:
            self.__lastAlpha = np.array(self.alpha) #保存上一次的混合系数
            self.__lastMu = np.array(self.mu) #保存上一次的均值
            self.__lastSigma = np.array(self.sigma) #保存上一次的协方差矩阵
            return False
        else:
            return True

    def train(self):
        print('gmm')
        times = 0
        while True:
            self.__eStep()    
            self.__mStep()
            times+=1
            print('迭代次数：',times)
            print('当前的似然函数值：',self.__likelihood())
            if self.__converged():
                break
        c = collections.defaultdict(list)
        for j in range(self.dataSize):
            c[np.argmax(self.gamma[j,:])].append(self.data[j])
        return c,self.mu
    
    def euclideanDistance(self,x,y):
        """
        计算欧式距离
        """
        return np.linalg.norm(x-y)

    def initializeMu(self):
        """
        从数据集中首先随机选择一个样本点作为初始均值向量，
        然后总是选择与当前样本最远的样本点作为下一个均值向量
        """
        self.mu = []
        self.mu.append(self.data[np.random.randint(0,self.dataSize)])
        for i in range(1,self.k):
            maxDistance = np.sum([self.euclideanDistance(self.data[0],self.mu[k]) for k in range(i)])
            temp = 0
            for j in range(self.dataSize):
                newMaxDistance = np.sum([self.euclideanDistance(self.data[j],self.mu[k]) for k in range(i)])
                if maxDistance < newMaxDistance:
                    maxDistance = newMaxDistance
                    temp = j
            self.mu.append(self.data[temp])