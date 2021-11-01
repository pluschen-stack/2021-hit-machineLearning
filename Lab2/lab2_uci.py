import numpy as np
from lab2 import newtonUCITest
def read_data(path):
    """
    读取数据
    """
    return np.loadtxt(path,str)

def processCNAE_Data(data):
    """
    处理数据，原本数据中有9个类别，这里将类别1和其他类别分开
    """
    dim = len(data[1].split(','))
    feature_data = np.ones((data.shape[0],dim))
    tag_data = np.zeros((data.shape[0],1))
    for i in range(len(data)):
        strlist = data[i].split(',')
        feature_data[i,1:] = strlist[1:]
        tag_data[i] = 1 if int(data[i].split(',')[0]) == 1 else 0
    return feature_data,tag_data,dim

def processHaberman_Data(data):
    dim = len(data[1].split(','))
    feature_data = np.ones((data.shape[0],dim))
    tag_data = np.zeros((data.shape[0],1))
    for i in range(len(data)):
        strlist = data[i].split(',')
        feature_data[i,1:] = strlist[:3]
        tag_data[i] = 1 if int(data[i].split(',')[3]) == 1 else 0
    return feature_data,tag_data,dim

def divide_data(feature_data,tag_data,ratio=0.3):
    trainsize = int(len(feature_data)*ratio)
    x_train_data = feature_data[:trainsize]
    x_test_data = feature_data[trainsize:]
    y_train_data = tag_data[:trainsize]
    y_test_data = tag_data[trainsize:]
    return x_train_data,x_test_data,y_train_data,y_test_data

if __name__ == '__main__':

    feature_data,tag_data,dim = processCNAE_Data(read_data('Lab2\CNAE-9.data'))
    x_train_data,x_test_data,y_train_data,y_test_data = divide_data(feature_data,tag_data,ratio=0.1)
    #不带惩罚项
    newtonUCITest(x_train_data,y_train_data,None,None,None,None,0,display = False,w = np.zeros((dim,1)),test = True,test_x=x_test_data,test_y=y_test_data)
    #带惩罚项
    newtonUCITest(x_train_data,y_train_data,None,None,None,None,np.exp(-1),display = False,w = np.zeros((dim,1)),test = True,test_x=x_test_data,test_y=y_test_data)

    feature_data,tag_data,dim = processHaberman_Data(read_data('Lab2\haberman.txt'))
    x_train_data,x_test_data,y_train_data,y_test_data = divide_data(feature_data,tag_data,ratio=2/360)
    #不带惩罚项
    newtonUCITest(x_train_data,y_train_data,None,None,None,None,0,display = False,w = np.zeros((dim,1)),test = True,test_x=x_test_data,test_y=y_test_data)
    #带惩罚项
    newtonUCITest(x_train_data,y_train_data,None,None,None,None,np.exp(-1),display = False,w = np.zeros((dim,1)),test = True,test_x=x_test_data,test_y=y_test_data)