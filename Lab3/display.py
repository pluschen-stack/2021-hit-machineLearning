import matplotlib.pyplot as plt
import numpy as np



plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def displayCompareRaw(rawData,c,clusterCentroids,title=''):
    """
    函数功能：展示聚类结果和原始数据对比
    """
    plt.style.use("seaborn")
    plt.subplot(211)
    plt.title('raw')
    displayRawData(rawData)
    plt.subplot(212)
    for i in range(len(c)):
        x = np.array(c[i])[:,0]
        y = np.array(c[i])[:,1]
        plt.scatter(x,y,marker=".", s=40)
        plt.scatter(clusterCentroids[i][0],clusterCentroids[i][1],marker='x',color='black')
    plt.title(title)
    plt.show()

def displayCompareResult(rawData,k,c1,c2,clusterCentroids1,clusterCentroids2,title1,title2):
    """
    函数功能:展现kmeans和gmm的对比结果
    """
    plt.style.use("seaborn")
    plt.subplot(311)
    plt.title('rawData')
    temp = 0
    sliceTemp = int(len(rawData)/k)
    postTemp = temp+sliceTemp
    for i in range(k):
        plt.scatter(rawData[temp:postTemp,:][:,0],rawData[temp:postTemp,:][:,1],marker='.',s=40)
        temp += sliceTemp
        postTemp = postTemp + sliceTemp
    plt.subplot(312)
    plt.title(title1)
    for i in range(len(c1)):
        x = np.array(c1[i])[:,0]
        y = np.array(c1[i])[:,1]
        plt.scatter(x,y,marker=".", s=40)
        plt.scatter(clusterCentroids1[i][0],clusterCentroids1[i][1],marker='x',s=250,color='black')
    plt.subplot(313)
    plt.title(title2)
    for i in range(len(c2)):
        x = np.array(c2[i])[:,0]
        y = np.array(c2[i])[:,1]
        plt.scatter(x,y,marker=".", s=40)
        plt.scatter(clusterCentroids2[i][0],clusterCentroids2[i][1],marker='x',s=250,color='black')
    plt.show()

def displayRawData(rawData,k):
    """
    函数功能：展示原始数据
    """
    plt.title('原始数据')
    temp = 0
    sliceTemp = int(len(rawData)/k)
    postTemp = temp+sliceTemp
    for i in range(k):
        plt.scatter(rawData[temp:postTemp,:][:,0],rawData[temp:postTemp,:][:,1],marker='.',s=40)
        temp += sliceTemp
        postTemp = postTemp + sliceTemp
    plt.show()