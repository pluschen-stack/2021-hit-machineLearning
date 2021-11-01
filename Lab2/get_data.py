import numpy as np
from display import displayResult
from display import displayMoreData


def generate_data(mean,cov_xy,var,tag,size=30):
    """
    生成指定的多元正态分布的数据
    mean：均值点
    cov_xy：协方差矩阵
    var：独立同分布的每个随机变量的方差
    size：数据量，是一个int类型数据
    tag：数据标签
    """
    cov = [[var,cov_xy],[cov_xy,var]]
    x = np.random.multivariate_normal(mean, cov, size)
    y = np.zeros(size)
    if tag == 1:
        y = np.ones(size)
    return x.reshape(size,2),y.reshape(size,1)

def get_data(mean1,mean2,cov_xy,var,size_pos,size_neg):
    """
    生成两组多元正态分布数据
    返回训练数据集：特征集和标签集
    同时返回：可以供打印的点(x1,x2)
    """
    k = []
    k.append(generate_data(mean1,cov_xy,var,0,size_pos))
    k.append(generate_data(mean2,cov_xy,var,1,size_neg))
    
    train_x = np.zeros((size_pos+size_neg,3))
    train_x[:,:1] = np.ones((size_neg+size_pos,1))
    train_x[:size_pos,1:] = k[0][0]
    train_x[size_pos:,1:] = k[1][0]
    train_y = np.zeros((size_neg+size_pos,1))
    train_y[:size_pos,:] = k[0][1]
    train_y[size_pos::] = k[1][1]
    
    x = []
    y = []
    x.append(k[0][0][:,0])
    y.append(k[0][0][:,1])
    x.append(k[1][0][:,0])
    y.append(k[1][0][:,1])
    return train_x,train_y,x,y


if __name__ == '__main__':
    cov_xy = 0.3
    var = 0.6
    k = []
    mean = np.array([2,1])
    k.append(generate_data(mean,cov_xy,var,50)[0])
    mean = np.array([-2,-1])
    k.append(generate_data(mean,cov_xy,var,50)[0])
    
    x = []
    y = []
    x.append(k[0][:,0])
    y.append(k[0][:,1])
    x.append(k[1][:,0])
    y.append(k[1][:,1])
    displayMoreData(x,y,title=' ',colors=['green','red'])
    displayResult([1,1,1],x,y,title=' ',colors=['green','red'])