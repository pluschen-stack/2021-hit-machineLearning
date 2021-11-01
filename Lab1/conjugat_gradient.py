import numpy as np

class ConjugatGradient(object):

    def __init__(self,x_matrix,y_tag,penalty,delta):
        self.x_matrix = x_matrix
        self.y_tag = y_tag
        self.penalty = penalty
        self.delta = delta
        
    def __switch(self,w_0):
        A = self.x_matrix.T @ self.x_matrix + self.penalty * np.eye(self.x_matrix.shape[1])
        x = w_0
        b = self.x_matrix.T @ self.y_tag
        return A,x,b

    def train(self,w_0):
        A,x,b = self.__switch(w_0)
        r = b - np.dot(A,x)
        p = r
        round = 0
        while True:
            if np.dot(p.T,p) < self.delta: 
                return round,x
            else:
                pre_w = x
                a = np.dot(r.T,r)/np.dot(np.dot(p.T,A),p)
                x = pre_w + a*p
                pre_r = r
                r = pre_r - a*np.dot(A,p)
                b = np.dot(r.T,r)/np.dot(pre_r.T,pre_r)
                p = r + b*p
                round += 1

