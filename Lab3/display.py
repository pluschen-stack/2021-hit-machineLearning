import matplotlib.pyplot as plt
import numpy as np



plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def displayCompareResult(rawData,c):
    """
    函数功能：展示kmeans结果和原始数据对比
    """
    plt.subplot(211)
    displayRawData(rawData)
    plt.subplot(212)
    for i in range(len(c)):
        x = np.array(c[i])[:,0]
        y = np.array(c[i])[:,1]
        plt.scatter(x,y,marker='*')
    plt.show()

def displayRawData(rawData):
    """
    函数功能：展示原始数据
    """
    plt.scatter(rawData[:,0],rawData[:,1],marker='*')