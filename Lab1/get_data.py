import numpy as np
import pandas as pd

def generate_data(begin,end,size,function,mu=0,sigma=0.3):
    """
    函数功能：产生数据并使用pandas的dataframe数据结构存储
    begin：样本中自变量起始下界
    end：样本中自变量终止上界
    size：样本数量的大小
    function:产生数据所使用的函数
    testratio:测试样本的比例
    mu：正态分布的均值
    sigma：正太分布的方差
    """
    x = np.linspace(begin,end,size)
    guass_noise = np.random.normal(mu,sigma,size)
    y = function(x)+guass_noise
    #np.dstack(x,y)得到的结果的shape是(1,320,2)的，所以要取第一项
    data = pd.DataFrame(np.dstack((x,y))[0],columns=['x','y'])
    return data

def divide_data(data,testratio=0.2):
    """
    函数功能：使用留出法将数据划分为训练数据和测试数据
    data：需要是dataframe类型的
    testratio:测试数据/总数据数
    """
    test_data = data.sample(frac=testratio,axis=0)
    train_data = pd.concat([data, test_data, test_data]).drop_duplicates(keep=False)
    return train_data,test_data.sort_index()

def get_data(begin,end,size,function,testratio):
    """
    函数功能：生成数据并划分，同时是得到数据格式一致
    """
    train_data,test_data = divide_data(generate_data(begin,end,size,function),testratio)
    x_train_data = np.array(train_data['x'].tolist())
    x_train_data = x_train_data.reshape(x_train_data.shape[0],1)
    y_train_data = np.array(train_data['y'].tolist())
    y_train_data = y_train_data.reshape(y_train_data.shape[0],1)
    x_test_data = np.array(test_data['x'].tolist())
    x_test_data = x_test_data.reshape(x_test_data.shape[0],1)
    y_test_data = np.array(test_data['y'].tolist())
    y_test_data = y_test_data.reshape(y_test_data.shape[0],1)
    return x_train_data,y_train_data,x_test_data,y_test_data

def get_x_matrix(x_vector,order=3):
    """
    函数功能：获得类似范德蒙德行列式的结构
    x_vector：自变量
    order：选择使用多少维度的模型，也就是多项式的最高指数
    """
    x_matrix = []
    for i in range(len(x_vector)):
        x_series = [1]
        for j in range(order):
            x_series.append(x_series[-1]*x_vector[i])
        x_matrix.append(x_series)  
    return np.array(x_matrix,dtype='float')