from getdata import generateData
from display import display2dim, display2dimWithResult, display3dim, display3dimWithResult
import numpy as np
from pca import PCA
from PIL import Image
import numpy as np
import os

def readData(fileName):
    """
    从指定fileName中读取图片并转化成对应的ndarray
    """
    im = Image.open('50x50_grey_anime_face\data\\{}'.format(fileName))
    data = np.array(im).reshape(2500)
    return data

def saveData(data,fileName):
    """
    将处理后的数据保存到新的文件夹
    """
    im = Image.fromarray(np.uint8(data))
    im = im.convert('L')
    im.save('50x50_grey_anime_face\\newData\\{}'.format(fileName))

def processData(targetdim):
    file_dir = "50x50_grey_anime_face\data"
    sumPSNR = 0
    rawData = []
    for root, dirs, files in os.walk(file_dir,topdown=True):
        for fileName in files:
            rawData.append(readData(fileName))
    rawData = np.array(rawData)
    model = PCA(rawData,targetdim)
    featureVectors, mean = model.train()
    lowDimData = model.transform(featureVectors)
    i = 0
    for root, dirs, files in os.walk(file_dir,topdown=True):
        for fileName in files:
            sumPSNR = calPSNR(rawData[i],lowDimData[i])
            saveData(lowDimData[i].reshape(50,50),fileName)
            i += 1
    print('总的信噪比：',sumPSNR)


def calPSNR(source,target):
    """
    计算图片峰值信噪比
    """
    diff = source - target
    diff = diff ** 2
    rmse = np.sqrt(np.mean(diff))
    return 20 * np.log10(255.0 / rmse)


if __name__ == '__main__':
    #二维数据
    mean = np.array([0,0])
    cov_xy = np.array([[1,0],[0,0.1]])
    data2d = generateData(mean,cov_xy,150)
    #三维数据
    mean = np.array([0,0,0])
    cov_xy = np.array([[1,0,0],[0,1,0],[0,0,0.1]])
    data3d = generateData(mean,cov_xy,150)
    #对于人脸数据集
    targetdim = 2

    model = PCA(data2d,1)
    featureVectors, mean = model.train()
    lowDimData = model.transform(featureVectors)
    display2dimWithResult(data2d,lowDimData)

    model = PCA(data3d,2)
    featureVectors, mean = model.train()
    lowDimData = model.transform(featureVectors)
    display3dimWithResult(data3d,lowDimData)

    processData(targetdim)