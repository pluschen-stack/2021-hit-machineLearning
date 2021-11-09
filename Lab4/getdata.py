import numpy as np

def generateData(mean,cov_xy,size):
    """
    mean:均值
    cov_xy:方差矩阵
    size:数目
    """
    data = np.random.multivariate_normal(mean, cov_xy, size)
    return data
