import matplotlib.pyplot as plt
import numpy as np



plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def displayCompareRaw(rawData,c,clusterCentroids,title=''):
    """
    函数功能：展示聚类结果和原始数据对比
    """
    plt.subplot(211)
    plt.title('raw')
    displayRawData(rawData)
    plt.subplot(212)
    for i in range(len(c)):
        x = np.array(c[i])[:,0]
        y = np.array(c[i])[:,1]
        plt.scatter(x,y,marker=".", s=40)
        plt.scatter(clusterCentroids[i][0],clusterCentroids[i][1],marker='X',color='black')
    plt.title(title)
    plt.show()

def displayCompareResult(c1,c2,clusterCentroids1,clusterCentroids2):
    """
    函数功能:展现kmeans和gmm的对比结果
    """
    plt.style.use("seaborn")
    plt.subplot(211)
    plt.title('kmeans')
    for i in range(len(c1)):
        x = np.array(c1[i])[:,0]
        y = np.array(c1[i])[:,1]
        plt.scatter(x,y,marker=".", s=40)
        plt.scatter(clusterCentroids1[i][0],clusterCentroids1[i][1],marker='X',color='black')
    plt.subplot(212)
    plt.title('gmm')
    for i in range(len(c2)):
        x = np.array(c2[i])[:,0]
        y = np.array(c2[i])[:,1]
        plt.scatter(x,y,marker=".", s=40)
        plt.scatter(clusterCentroids2[i][0],clusterCentroids2[i][1],marker='X',color='black')
    plt.show()

def displayRawData(rawData):
    """
    函数功能：展示原始数据
    """
    plt.scatter(rawData[:,0],rawData[:,1])