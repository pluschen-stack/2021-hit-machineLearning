import numpy as np
import random
import collections

"""
kmeans方法划分聚类，如果选择聚类数过多而样本难以分出这么多类就会出现错误，因为这里没有考虑丢弃k
"""
class KMeans(object):
    def __init__(self,data,k,delta = 1e-7) -> None:
        """
        data：数据
        k：聚类数
        """
        self.data = data
        self.k = k
        self.delta = delta
        self.tag = np.zeros(self.data.shape[0])
        self.dataSize = len(data)

    def euclideanDistance(self,x,y):
        """
        计算欧式距离
        """
        return np.linalg.norm(x-y)
        
    def train(self):
        """
        训练
        """
        print('kmeans')
        times = 0
        c = collections.defaultdict(list)
        while True:
            c = collections.defaultdict(list)
            for i in range(self.dataSize):
                self.tag[i] = np.argmin([self.euclideanDistance(self.data[i],self.clusterCentroids[j]) for j in range(self.k)])
                c[self.tag[i]].append(self.data[i])
            newClusterCentroids = [np.mean(c[i],axis=0) for i in range(self.k)]
            times += 1
            print('迭代次数：',times)
            if self.euclideanDistance(np.array(self.clusterCentroids),np.array(newClusterCentroids)) < self.delta:
                break
            else:
                self.clusterCentroids = newClusterCentroids
        return c,self.clusterCentroids,self.tag

    def initializeRemoteK(self):
        """
        从数据集中首先随机选择一个样本点作为初始均值向量，
        然后总是选择与当前样本最远的样本点作为下一个均值向量
        """
        self.clusterCentroids = []
        self.clusterCentroids.append(self.data[np.random.randint(0,self.dataSize)])
        for i in range(1,self.k):
            maxDistance = np.sum([self.euclideanDistance(self.data[0],self.clusterCentroids[k]) for k in range(i)])
            temp = 0
            for j in range(self.dataSize):
                newMaxDistance = np.sum([self.euclideanDistance(self.data[j],self.clusterCentroids[k]) for k in range(i)])
                if maxDistance < newMaxDistance:
                    maxDistance = newMaxDistance
                    temp = j
            self.clusterCentroids.append(self.data[temp])
        return self.train()

    def randomInitializeK(self):
        """
        从数据集中随机选择k个样本作为初始均值向量
        这个总是出现nan问题不好用，不用这个了
        """
        self.clusterCentroids = np.array(random.sample(self.data.tolist(),self.k))
        return self.train()
            

if __name__ == '__main__':
    """
    """