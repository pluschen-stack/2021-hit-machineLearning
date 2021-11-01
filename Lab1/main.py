from conjugat_gradient import ConjugatGradient
from gradient_descent import *
from get_data import * 
from analytical_solution import *
from display import *

if __name__ == '__main__':

    begin = 0
    end = 1
    size = 50
    order = 8
    testratio = 0.2
    penalty_lambda = np.exp(-12)
    alpha = 0.02
    xlimit=(begin,end)
    ylimit=(-1,1)

    x_train_data,y_train_data,x_test_data,y_test_data = get_data(begin,end,size,lambda x :np.sin(2*np.pi*x),testratio)
    x_matrix = get_x_matrix(x_train_data,order)
    x_test_matrix = get_x_matrix(x_test_data,order)

    # 查找比较好lambda
    #show_Erms(x_train_data,y_train_data,x_test_data,y_test_data,order,title='查看测试数据集和训练数据集上的损失',penalty_lambda=range(-60,0))

    # #numpy自带的多项式拟合
    # w = np.polyfit(x_train_data.reshape(x_train_data.shape[0]),y_train_data.reshape(y_train_data.shape[0]),order)
    # w = w[::-1]
    # show_compared_to_data(w,x_test_data,y_test_data,'numpy自带的多项式拟合',lambda x:np.sin(2*np.pi*x),xlimit,ylimit,order)
    # show_compared_to_data(w,x_train_data,y_train_data,'numpy自带的多项式拟合',lambda x:np.sin(2*np.pi*x),xlimit,ylimit,order)
    
    #开始训练不带惩罚项的解析解  
    train = AnalyticalSolution(x_matrix,y_train_data)
    w0 = train.normal()
    #训练结果
    show_compared_to_data(w0,x_test_data,y_test_data,'解析解，和测试数据对比,阶数={0},样本数={1}'.format(order,size*testratio),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)
    print('训练集loss = {0},测试集loss = {1},order = {2}'.format(train.normal_loss(),train.normal_loss_test(x_test_matrix,y_test_data),order))
    show_compared_to_data(w0,x_train_data,y_train_data,'解析解，和训练数据对比,阶数={0},样本数={1}'.format(order,size-size*testratio),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)


    #开始训练带惩罚项的解析解  
    train = AnalyticalSolution(x_matrix,y_train_data)
    w1 = train.with_penalty(penalty_lambda)
    # #训练结果
    # show_compared_to_data(w1,x_test_data,y_test_data,'解析解，和测试数据对比,阶数={0},样本数={1},惩罚项={2}'.format(order,size*testratio,penalty_lambda),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)
    # print(train.with_penalty_loss())
    # show_compared_to_data(w1,x_train_data,y_train_data,'解析解，和训练数据对比,阶数={0},样本数={1},惩罚项={2}'.format(order,size-size*testratio,penalty_lambda),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)

    #开始梯度下降训练
    model = GradientDescent(x_matrix,y_train_data,penalty_lambda,alpha)
    w2 = model.train()
    # print('梯度下降运行了{0}轮'.format(len(w2[0])))
    # show_compared_to_data(w2[1],x_test_data,y_test_data,'梯度下降，和测试数据对比,阶数={0},样本数={1},惩罚项={2},学习率={3}'.format(order,size*testratio,penalty_lambda,alpha),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)
    # show_compared_to_data(w2[1],x_train_data,y_train_data,'梯度下降，和训练数据对比,阶数={0},样本数={1},惩罚项={2},学习率={3}'.format(order,size-size*testratio,penalty_lambda,alpha),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)

    #开始共轭梯度下降训练
    model = ConjugatGradient(x_matrix,y_train_data,penalty_lambda,1e-6)
    w_0 = np.zeros(x_matrix.shape[1]).reshape(x_matrix.shape[1],1)
    w3 = model.train(w_0)
    # print('共轭梯度下降运行了{0}轮'.format(w3[0]))
    # #训练结果
    # show_compared_to_data(w3[1],x_test_data,y_test_data,'共轭梯度下降，和测试数据对比,阶数={0},样本数={1},惩罚项={2}'.format(order,size*testratio,penalty_lambda),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)
    # show_compared_to_data(w3[1],x_train_data,y_train_data,'共轭梯度下降，和训练数据对比,阶数={0},样本数={1},惩罚项={2}'.format(order,size-size*testratio,penalty_lambda),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)
    

    # #尝试多阶数不带惩罚项的解析解
    # for order in range(20):
    #     #开始训练不带惩罚项的解析解  
    #     x_matrix = get_x_matrix(x_train_data,order)
    #     x_test_matrix = get_x_matrix(x_test_data,order)
    #     train = AnalyticalSolution(x_matrix,y_train_data)
    #     w = train.normal()
    #     #训练结果
    #     # show_compared_to_data(w,x_test_data,y_test_data,'解析解，和测试数据对比,阶数={0},样本数={1}'.format(order,size*testratio),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order)
    #     print('训练集loss = {0},测试集loss = {1},order = {2}'.format(train.normal_loss(),train.normal_loss_test(x_test_matrix,y_test_data),order))
    #     # show_compared_to_data(w,x_train_data,y_train_data,'解析解，和训练数据对比,阶数={0},样本数={1}'.format(order,size-size*testratio),lambda x :np.sin(2*np.pi*x),xlimit,ylimit,order) 

    # #尝试不同的学习率的梯度下降
    # show_alpha(x_train_data,y_train_data,alpha=[0.1,0.11,0.12],order=order,title='不同的学习率之间的对比,阶数={0},训练样本数={1}'.format(order,x_train_data.size),penalty_lambda=penalty_lambda,labels=['0.1','0.11','0.12'],colors=['blue','green','red'])

    # #梯度下降方法和带惩罚项的解析解答案对比
    # show_comparasion(w=[w1,w2[1]],x=x_train_data,y=y_train_data,title='梯度下降方法和带惩罚项的解析解答案在训练集对比,阶数={0},训练样本数={1}'.format(order,x_train_data.size),colors=['blue','red'],labels=['带惩罚项的解析解','梯度下降'],function = lambda x :np.sin(2*np.pi*x),xlimit=(0,1),ylimit=(-1,1),order=order)
    # show_comparasion(w=[w1,w2[1]],x=x_test_data,y=y_test_data,title='梯度下降方法和带惩罚项的解析解答案在测试集对比,阶数={0},训练样本数={1}'.format(order,x_train_data.size),colors=['blue','red'],labels=['带惩罚项的解析解','梯度下降'],function = lambda x :np.sin(2*np.pi*x),xlimit=(0,1),ylimit=(-1,1),order=order)
    
    # #梯度下降方法中对于不同的阶数大小对应的轮数
    # N = range(1,12)
    # rounds = []
    # for n in N:
    #     x_train_data,y_train_data,x_test_data,y_test_data = get_data(begin,end,size,lambda x :np.sin(2*np.pi*x),testratio)
    #     x_matrix = get_x_matrix(x_train_data,n)
    #     model = GradientDescent(x_matrix,y_train_data,penalty_lambda,alpha)
    #     w2 = model.train()
    #     if np.isnan(w2[1]).any() :
    #         rounds.append(0)
    #     else:
    #         rounds.append(len(w2[0]))
    # plt.plot(N,rounds)
    # plt.xlabel('order')
    # plt.ylabel('rounds')
    # plt.title('梯度下降方法中对于不同的阶数大小对应的轮数')
    # plt.show()

    # #共轭梯度下降方法中对于不同的阶数大小对应的轮数
    # N = range(1,12)
    # rounds = []
    # for n in N:
    #     x_train_data,y_train_data,x_test_data,y_test_data = get_data(begin,end,size,lambda x :np.sin(2*np.pi*x),testratio)
    #     x_matrix = get_x_matrix(x_train_data,n)
    #     model = ConjugatGradient(x_matrix,y_train_data,penalty_lambda,1e-6)
    #     w_0 = np.zeros(x_matrix.shape[1]).reshape(x_matrix.shape[1],1)
    #     w3 = model.train(w_0)
    #     if np.isnan(w3[1]).any() :
    #         rounds.append(0)
    #     else:
    #         rounds.append(w3[0])
    # plt.plot(N,rounds)
    # plt.xlabel('order')
    # plt.ylabel('rounds')
    # plt.title('共轭梯度下降方法中对于不同的阶数大小对应的轮数')
    # plt.show()

    # #共轭梯度下降方法中对于不同的样本数大小对应的轮数
    # N = range(10,500,10)
    # rounds = []
    # for n in N:
    #     x_train_data,y_train_data,x_test_data,y_test_data = get_data(begin,end,n,lambda x :np.sin(2*np.pi*x),testratio)
    #     x_matrix = get_x_matrix(x_train_data,order)
    #     model = ConjugatGradient(x_matrix,y_train_data,penalty_lambda,1e-6)
    #     w_0 = np.zeros(x_matrix.shape[1]).reshape(x_matrix.shape[1],1)
    #     w3 = model.train(w_0)
    #     if np.isnan(w3[1]).any() :
    #         rounds.append(0)
    #     else:
    #         rounds.append(w3[0])
    # plt.plot(N,rounds)
    # plt.xlabel('N')
    # plt.ylabel('rounds')
    # plt.title('共轭梯度下降方法中对于不同的样本数大小对应的轮数')
    # plt.show()

    #对4个方法的效果进行对比
    show_comparasion([w0,w1,w2[1],w3[1]],x_test_data,y_test_data,title='对4个方法的效果在测试集进行对比,阶数={0},训练样本数={1}'.format(order,x_train_data.size),function=lambda x:np.sin(2*np.pi*x),colors=['blue','red','yellow','purple'],labels=['解析解','带惩罚项的解析解','梯度下降','共轭梯度下降'],order=order)
    show_comparasion([w0,w1,w2[1],w3[1]],x_train_data,y_train_data,title='对4个方法的效果在训练集进行对比,阶数={0},训练样本数={1}'.format(order,x_train_data.size),function=lambda x:np.sin(2*np.pi*x),colors=['blue','red','yellow','purple'],labels=['解析解','带惩罚项的解析解','梯度下降','共轭梯度下降'],order=order)

    