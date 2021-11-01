import numpy as np
import matplotlib.pyplot as plt

def generateData(mean,cov_xy,var,K=4,size=30):
    """
    生成指定的二元正态分布的数据
    mean：均值点
    cov_xy：协方差
    var：独立同分布的每个随机变量的方差
    K: 生成的样本的类数
    size：数据量，是一个int类型数据
    """
    cov = [[var,cov_xy],[cov_xy,var]]
    sampledata = []
    for i in range(len(mean)):
        sampledata.append(np.random.multivariate_normal(mean[i], cov, size))
    return np.array(sampledata).reshape(size*K,2)

if __name__ == '__main__':
    mean = [np.array((1,1)),np.array((4,4)),np.array((1,4)),np.array((4,1))]
    data = generateData(mean,0,1,len(mean))
    plt.scatter(data[:,0],data[:,1],'black')
    plt.show()
