import numpy as np
from display import displayCompareResult
from getdata import generateData
from kmeans import KMeans


if __name__ == '__main__':
    mean = [np.array((1,1)),np.array((4,4)),np.array((1,4)),np.array((4,1))]

    data = generateData(mean,0,1,len(mean))
    model = KMeans(data,4)
    
    c,clusterCentroids = model.InitializeRemoteK()
    displayCompareResult(data,c)
