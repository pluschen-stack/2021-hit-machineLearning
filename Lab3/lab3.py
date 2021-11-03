from itertools import permutations
import numpy as np
from display import displayCompareResult,displayCompareRaw, displayRawData
from getdata import generateData
from gmm import GMM
from kmeans import KMeans
import pandas as pd

def readIrisData(fileName):
    df = pd.DataFrame(pd.read_csv(fileName))
    data = df.iloc[:,0:4].to_numpy()
    tag = []
    for i in range(len(df)):
        if df.iloc[i,4] == 'Iris-setosa':
            tag.append(0)
        elif df.iloc[i,4] == 'Iris-versicolor':
            tag.append(1)
        elif df.iloc[i,4] == 'Iris-virginica':
            tag.append(2)
    return data,tag

def readSeedsData(fileName):
    df = pd.DataFrame(pd.read_csv(fileName,delimiter='\t'))
    data = df.iloc[:,0:7].to_numpy()
    tag = df.iloc[:,7].to_numpy()
    return data,tag

def readUserModelingData(fileName):
    df = pd.DataFrame(pd.read_csv(fileName,delimiter='\t'))
    data = df.iloc[:,0:5].to_numpy()
    tag = []
    for i in range(len(df)):
        if df.iloc[i,5] == 'High':
            tag.append(0)
        elif df.iloc[i,5] == 'Low':
            tag.append(1)
        elif df.iloc[i,5] == 'very_low':
            tag.append(2)
        elif df.iloc[i,5] == 'Middle':
            tag.append(3)
    return data,tag
def accuracy(realLabel,predictLabel,k):
    """
    使用全排列的方式计算聚类准确率
    """
    classes = list(permutations(range(k), k))
    counts = np.zeros(len(classes))
    for i in range(len(classes)):
        for j in range(realLabel.shape[0]):
            if int(realLabel[j]) == classes[i][int(predictLabel[j])]:
                counts[i] += 1
    return np.max(counts) / realLabel.shape[0]

def myTest(data,tag,k):
    model = KMeans(data,k)
    c1,clusterCentroids1,tag1 = model.initializeRemoteK()
    # displayCompareRaw(data,c1,clusterCentroids1,title='Kmeans,accuracy={}'.format(accuracy(np.array(tag),tag1,len(mean))))

    model = GMM(data,k)
    c2,clusterCentroids2,tag2 = model.train()
    # displayCompareRaw(data,c2,clusterCentroids2,title='GMM,accuracy={}'.format(accuracy(np.array(tag),tag2,len(mean))))
    displayCompareResult(data,k,c1,c2,clusterCentroids1,clusterCentroids2,title1 ='Kmeans',title2 = 'GMM')

if __name__ == '__main__':
    # 符合kmeans模型的样本
    x = 2
    mean = [np.array((-x,-x)),np.array((x,x)),np.array((-x,x)),np.array((x,-x))]
    size = [80,80,80,80]
    data = generateData(mean,0.6,2,size,len(mean))
    tag = [int(i/80) for i in range(sum(size))]
    myTest(data,tag,len(mean))
    
    # 符合gmm模型的样本
    mean = [np.array((1,3)),np.array((2,2)),np.array((3,1))]
    size = [120,120,120]
    data = generateData(mean,0.6,1,size,len(mean))
    tag = [int(i/80) for i in range(sum(size))]
    myTest(data,tag,3)

    # #鸢尾花数据集聚类
    # irisData,irisTag = readIrisData('iris.csv')
    # model = KMeans(irisData,3)
    # c1,clusterCentroids1,tag1 = model.initializeRemoteK()
    # print('Kmeans acurracy in iris：',accuracy(np.array(irisTag),tag1,3))
    # model = GMM(irisData,3)
    # c2,clusterCentroids2,tag2 = model.train()
    # print('GMM acurracy in iris：',accuracy(np.array(irisTag),tag2,3))

    # #种子数据集
    # data,tag = readSeedsData('seeds_dataset.txt')
    # print(data,tag)
    # model = KMeans(data,3)
    # c1,clusterCentroids1,tag1 = model.initializeRemoteK()
    # print('Kmeans acurracy in  seedsData：',accuracy(np.array(tag),tag1,3))
    # model = GMM(data,3)
    # c2,clusterCentroids2,tag2 = model.train()
    # print('GMM acurracy in seedsData：',accuracy(np.array(tag),tag2,3))

    # #DataUserModelingData
    # data,tag = readUserModelingData('Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.csv')
    # print(np.array(data).shape,np.array(tag).shape)
    # model = KMeans(data,4)
    # c1,clusterCentroids1,tag1 = model.initializeRemoteK()
    # print('Kmeans acurracy in usermodelingdata：',accuracy(np.array(tag),tag1,4))
    # model = GMM(data,4)
    # c2,clusterCentroids2,tag2 = model.train()
    # print('GMM acurracy in usermodelingdata：',accuracy(np.array(tag),tag2,4))