import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def display3dim(data):
    """
    需要一个n*3的矩阵
    """
    ax = plt.axes(projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2],c=data[:,2])
    plt.show()


def display2dim(data):
    """
    需要一个n*2的矩阵
    """
    ax = plt.axes()
    ax.scatter(data[:,0],data[:,1])
    plt.show()

def display2dimWithResult(rawdata,transformedData):
    """
    在2d的情况下对比结果
    """
    ax = plt.axes()
    ax.scatter(rawdata[:,0],rawdata[:,1],label='原始数据')
    ax.scatter(transformedData[:,0],transformedData[:,1],label='降维后的数据')
    plt.legend()
    plt.show()

def display3dimWithResult(rawdata,transformedData):
    """
    在3d的情况下对比结果
    """
    ax = plt.axes(projection='3d')
    ax.scatter(rawdata[:,0],rawdata[:,1],rawdata[:,2],label='原始数据')
    ax.scatter(transformedData[:,0],transformedData[:,1],transformedData[:,2],label='降维后的数据')
    plt.legend()
    plt.show()
