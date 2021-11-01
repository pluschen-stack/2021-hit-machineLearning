import numpy as np
from new_ton import NewTon
from get_data import get_data
from display import displayResult
from gradient_descent import GradientDescent

def accuracy(w,X,Y):
    """
    计算正确率
    P(Y=0|X,W)=1/(1+exp(w0+sum(wiXi)))
    """
    predict_Y = np.zeros((X.shape[0],1))
    for i in range(len(X)):
        if X[i] @ w > 0 :
            predict_Y[i] = 1
        else:
            predict_Y[i] = 0
    boolY = predict_Y == Y
    return np.sum(boolY)/boolY.shape[0]
            
def test(test_x,test_y,train_x,train_y,x,y,title,colors,alpha,lambda_penalty,display =True,w = np.zeros(3)):
    """
    对模型测试并打印
    将会输出模型，模型的模值，损失值，准确率，以及展示图
    """
    model = GradientDescent(w,train_x,train_y,alpha,lambda_penalty=lambda_penalty)
    w = model.train()
    print('w={0}，它的模大小{1}'.format(w,w.T @ w))
    print('在训练集loss{0}'.format(model.loss(train_x,train_y)))
    print('在测试集loss{0}'.format(model.loss(test_x,test_y)))
    print('在训练集准确率{0}'.format(accuracy(w,train_x,train_y)))
    print('在测试集准确率{0}'.format(accuracy(w,test_x,test_y)))
    if display:
        displayResult(w,x,y,title,colors)
        

def newtonTest(test_x,test_y,train_x,train_y,x,y,title,colors,lambda_penalty,display =True,w = np.zeros(3)):
    """
    对模型测试并打印
    将会输出模型，模型的模值，损失值，准确率，以及展示图
    """
    model = NewTon(w,train_x,train_y,lambda_penalty=lambda_penalty)
    w = model.train()
    print('w={0}，它的模大小{1}'.format(w,w.T @ w))
    print('在训练集loss{0}'.format(model.loss(train_x,train_y)))
    print('在测试集loss{0}'.format(model.loss(test_x,test_y)))
    print('在训练集准确率{0}'.format(accuracy(w,train_x,train_y)))
    print('在测试集准确率{0}'.format(accuracy(w,test_x,test_y)))
    if display:
        displayResult(w,x,y,title,colors)

def newtonUCITest(train_x,train_y,x,y,title,colors,lambda_penalty,display =True,w = np.zeros(3),test = False,test_x=None,test_y=None):
    """
    对模型测试并且，如果数据是二维的就打印,否则无法打印
    将会输出模型，模型的模值，损失值，准确率，以及展示图
    """
    model = NewTon(w,train_x,train_y,lambda_penalty=lambda_penalty)
    w = model.train()
    print('它的模大小{}'.format(w.T @ w))
    print('train_loss{0}'.format(model.loss(train_x,train_y)))
    print('在训练集准确率{0}'.format(accuracy(w,train_x,train_y)))
    if display:
        displayResult(w,x,y,title,colors)
    if test:
        print('test_loss{0}'.format(model.loss(test_x,test_y)))
        print('在测试集准确率{0}'.format(accuracy(w,test_x,test_y)))

if __name__ == '__main__':
    mean1 = np.array([1,2]) #分类一的均值
    mean2 = np.array([-1,-2]) #分类二的均值
    cov_xy = 0 #协方差
    var = 1 #方差
    order = 3
    size_pos = 50 #反例数量
    size_neg = 50 #正例数量
    colors=['green','red']#正反例颜色
    penalty = np.exp(-1)

    # train_x,train_y,x,y = get_data(mean1,mean2,cov_xy,var,size_pos,size_neg)
    # test_x,test_y,x,y = get_data(mean1,mean2,cov_xy,var,size_pos*5,size_neg*5)
    # test(test_x,test_y,train_x,train_y,x,y,'满足朴素贝叶斯的无正则项',colors=['green','red'],alpha=0.03,lambda_penalty=0)
    # test(test_x,test_y,train_x,train_y,x,y,'满足朴素贝叶斯的正则项为{0}'.format(penalty),colors=['green','red'],alpha=0.03,lambda_penalty=penalty)
    # #---------牛顿法-------------
    # newtonTest(test_x,test_y,train_x,train_y,x,y,'满足朴素贝叶斯的无正则项的牛顿法',colors=['green','red'],lambda_penalty=0)
    # newtonTest(test_x,test_y,train_x,train_y,x,y,'满足朴素贝叶斯的正则项为{0}的牛顿法'.format(penalty),colors=['green','red'],lambda_penalty=penalty)
    #---------不满足朴素贝叶斯------------------------------
    cov_xy = 0.3
    train_x,train_y,x,y = get_data(mean1,mean2,cov_xy,var,size_pos,size_neg)
    test_x,test_y,x,y = get_data(mean1,mean2,cov_xy,var,size_pos*5,size_neg*5)
    test(test_x,test_y,train_x,train_y,x,y,'协方差系数为{},无正则项'.format(cov_xy),colors=['green','red'],alpha=0.03,lambda_penalty=0)
    test(test_x,test_y,train_x,train_y,x,y,'协方差系数为{},正则项为{}'.format(cov_xy,penalty),colors=['green','red'],alpha=0.03,lambda_penalty=penalty)
    #---------牛顿法-------------
    newtonTest(test_x,test_y,train_x,train_y,x,y,'协方差系数为{},无正则项的牛顿法'.format(cov_xy),colors=['green','red'],lambda_penalty=0)
    newtonTest(test_x,test_y,train_x,train_y,x,y,'协方差系数为{},正则项为{}的牛顿法'.format(cov_xy,penalty),colors=['green','red'],lambda_penalty=penalty)
    
    
    
