import matplotlib.pyplot as plt
import numpy as np



plt.rcParams['axes.facecolor']='snow'
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def displayData(x,y,color):
    """
    函数功能：展示数据
    """
    plt.scatter(x,y,c=color)

def displayMoreData(x,y,title,colors):
    """
    函数功能：展示多组数据
    """
    for i in range(len(x)):
        displayData(x[i],y[i],color=colors[i])
    plt.title(title)
    plt.grid()  
    plt.show()

def displayResult(w,x,y,title,colors,):
    """
    函数功能：展示两组数据以及数据之间的分界线
    """
    minx = min(x[0])
    maxx = max(x[0])
    for i in range(len(x)):
        displayData(x[i],y[i],color=colors[i])
        minx = min(minx,min(x[i]))
        maxx = max(maxx,max(x[i]))
    xlim = np.arange(minx,maxx,0.1)
    plt.plot(xlim,-w[1]/w[2]*xlim-w[0]/w[2],c='black')
    plt.title(title)
    plt.grid()  
    plt.show()
