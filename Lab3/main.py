import numpy as np
from display import displayCompareResult,displayCompareRaw
from getdata import generateData
from gmm import GMM
from kmeans import KMeans


if __name__ == '__main__':
    x = 2
    mean = [np.array((-x,-x)),np.array((x,x)),np.array((-x,x)),np.array((x,-x))]
    data = generateData(mean,0,1,len(mean))

    model = KMeans(data,len(mean))
    c1,clusterCentroids1 = model.initializeRemoteK()
    # displayCompareRaw(data,c1,clusterCentroids1,title='Kmeans')

    model = GMM(data,len(mean))
    c2,clusterCentroids2 = model.train()
    # displayCompareRaw(data,c2,clusterCentroids2,title='GMM')

    displayCompareResult(c1,c2,clusterCentroids1,clusterCentroids2)



