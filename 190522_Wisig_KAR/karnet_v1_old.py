import numpy as np

class KARnet:
    def __init__(self,data,h,rseed):
        self.x = data[0]
        self.y = data[1]
        self.xte = data[2]
        self.yte = data[3]
        
        self.bias = np.ones([self.x.shape[0],1])
        self.X = np.hstack([self.bias,self.x])
        self.rseed = rseed
        self.h = h
        self.W = []
        
    def f(self,x): # activation func
        return(np.arctan(x)) # arctan
        #return(np.log(x/(1-x))) # logit
    def finv(self,x): # inverse of activation func
        return(np.tan(x)) # tan
        #return(1/(1+np.exp(-x))) # sigmoid
        
    
    def pre_eq(self,X,bias,W):
        return 1*(np.hstack([bias,self.f(1*X.dot(W))]))
    def sur_eq(self,y,bias,v,w):
        return (self.finv(y)-1*bias.dot(v)).dot(1*np.linalg.pinv(w))

    def train(self):
        # initialize w
        np.random.seed(self.rseed)
        w,v = [],[]
        for m in np.arange(len(self.h)):
            w_row = self.h[len(self.h)-1-m]
            if m==0:
                w_col = self.y.shape[1]
            else:
                w_col = self.h[len(self.h)-m]

            w.append(np.random.rand(w_row,w_col))
            v.append(np.random.rand(1,w_col))   
        w.reverse()
        v.reverse()
        
        # get weight
        for pre_num in range(len(self.h)+1):
            sur_num = len(self.h) - 1
            sur_calc = self.y
            for s in np.arange(sur_num,pre_num-1,-1):
                sur_calc = self.sur_eq(sur_calc,self.bias,v[s],w[s])
            pre_calc = self.X
            for p in np.arange(0,pre_num,1):
                pre_calc = self.pre_eq(pre_calc,self.bias,self.W[p])
                
            W_calc = np.linalg.pinv(pre_calc).dot(self.finv(sur_calc))
            self.W.append(W_calc)
    
    # calc output
    def output(self,x):
        bias = np.ones([x.shape[0],1])
        x_calc = x
        for i in range(len(self.W)):
            x_calc = 1*self.f(np.hstack([bias,x_calc]).dot(self.W[i]))
        return(x_calc)
    
    # get accuracy
    def accuracy(self,mode='test'):
        if mode=='train':
            xte = self.x
            yte = self.y
        else:
            xte = self.xte
            yte = self.yte
            
        pred = self.output(xte)
        tf = [np.argmax(pred[i])==np.argmax(yte[i]) for i in range(len(pred))]
        acc = np.sum(tf) / len(tf)
        return(acc)