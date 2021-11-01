import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import GradientDescent
from analytical_solution import AnalyticalSolution
from get_data import get_x_matrix

plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def show_compared_to_data(w,x,y,title,function,xlimit=(0,1),ylimit=(-1,1),order=3):
    """
    函数功能：用于模型和给的数据之间的对比
    w:多项式参数，也就是训练好后的模型
    x:测试样本的自变量
    y:目标值
    xlimit和ylimit：自变量和因变量的显示范围
    order:模型阶数
    """
    plt.title(title)
    plt.scatter(x,y,c='red',label='噪声数据')
    plt.plot(x,np.dot(get_x_matrix(x,order),w),c='blue',label='模型')
    x=np.arange(xlimit[0],xlimit[1],0.01)
    y=function(x)
    plt.plot(x,y,c='lightgreen',label='y=sin(x)')
    plt.xlim(xlimit)
    plt.ylim(ylimit)
    plt.legend()
    plt.grid()  
    plt.show()

def show_comparasion(w,x,y,title,colors,labels,function,xlimit=(0,1),ylimit=(-1,1),order=3):
    """
    函数功能：用于多个模型之间的对比
    w:多项式参数，也就是训练好后的模型，可以是多个
    x:测试样本的自变量
    y:目标值
    colors：每个模型应该采用的颜色标记
    labels：每个模型的文字标记
    xlimit和ylimit：自变量和因变量的显示范围
    order:模型阶数
    """
    assert len(w) == len(labels) 
    assert len(labels) == len(colors)
    plt.title(title)
    for i in range(len(w)):
        plt.plot(x,np.dot(get_x_matrix(x,order),w[i]),c=colors[i],label=labels[i])
    x=np.arange(xlimit[0],xlimit[1],0.01)
    y=function(x)
    plt.plot(x,y,c='lightgreen',label='y=sin(x)')
    plt.xlim(xlimit)
    plt.ylim(ylimit)
    plt.legend()
    plt.grid()  
    plt.show()

def show_Erms(x_train_data,y_train_data,x_test_data,y_test_data,order,title,penalty_lambda):
    """
    函数功能：分别打印在数据集和训练集上带惩罚项解析解的Erms
    """
    plt.title(title)
    x_train_matrix = get_x_matrix(x_train_data,order)
    x_test_matrix = get_x_matrix(x_test_data,order)
    train_E_rms = []
    test_E_rms = []
    for l in penalty_lambda:
        e = np.exp(l)
        train = AnalyticalSolution(x_train_matrix,y_train_data)
        w = train.with_penalty(e)
        train_E_rms.append(train.E_rms(y_train_data,np.dot(x_train_matrix,w)))
        test_E_rms.append(train.E_rms(y_test_data,np.dot(x_test_matrix,w)))
    plt.xlabel('lnλ')
    plt.ylabel('Erms')
    plt.plot(penalty_lambda,train_E_rms,label='Erms on train set')
    plt.plot(penalty_lambda,test_E_rms,label='Erms on test set')
    plt.legend()
    plt.grid()
    plt.show()

def show_alpha(x_train_data,y_train_data,alpha,order,title,colors,labels,penalty_lambda):
    """
    函数功能：对梯度下降方法，对于不同的学习率展示比较曲线
    """
    plt.title(title)
    x_train_matrix = get_x_matrix(x_train_data,order)

    for i in range(len(alpha)):
        model = GradientDescent(x_train_matrix,y_train_data,penalty_lambda,alpha=alpha[i])
        w = model.train()
        print('梯度下降运行了{0}轮'.format(len(w[0])))
        plt.plot(w[0].reshape(w[0].shape[0],1),w[2].reshape(w[2].shape[0],1),color=colors[i],label=labels[i])
    plt.xlabel('rounds')
    plt.ylabel('Erms')
    plt.legend()
    plt.grid()
    plt.show()

