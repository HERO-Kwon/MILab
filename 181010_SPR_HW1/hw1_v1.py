##########################
## 1-2 SPR HW 1         ##
## Made by: HERO Kwon   ##
## Date : 20181010      ##
##########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#(iii) compute bayesian decision
def NormalDens(x, mean, std):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
    return (1 / (math.sqrt(2*math.pi) * std)) * exponent
def Prob_d(x,train_data,cls,var):
    train_mean = train_data.groupby(['d3']).mean()
    train_std = train_data.groupby(['d3']).std()
    return NormalDens(x,train_mean[var].loc[cls],train_std[var].loc[cls])
def BayesDecision(df_row,train_data):
    p_cls0 = 0.5*Prob_d(df_row['d1'],train_data,0,'d1')*Prob_d(df_row['d2'],train_data,0,'d2') 
    p_cls1 = 0.5*Prob_d(df_row['d1'],train_data,1,'d1')*Prob_d(df_row['d2'],train_data,1,'d2')
    return int(p_cls0<p_cls1)

#(v) cls error rate
def ClsError(df,test_df):
    pred_y = [BayesDecision(test_df.iloc[i],df) for i in range(len(test_df))]
    true_y= list(test_df['d3'].values.astype('int'))
    from sklearn.metrics import confusion_matrix
    cls_result = confusion_matrix(true_y,pred_y)
    cls_err = round(1 - np.sum(np.diag(cls_result)) / np.sum(cls_result),3)
    return [cls_result,cls_err]

#(vii) decision boundary
def DecisionBoundary(df):
    mean0 = np.mean(df[df['d3']==0][['d1','d2']]).values.reshape(2,1)
    mean1 = np.mean(df[df['d3']==1][['d1','d2']]).values.reshape(2,1)
    cov0 = np.cov(np.mat(df[df['d3']==0][['d1','d2']]).T)
    cov1 = np.cov(np.mat(df[df['d3']==1][['d1','d2']]).T)
    invcov0 = np.linalg.inv(cov0)
    invcov1 = np.linalg.inv(cov1)
    big_w0 = -0.5*invcov0
    w0 = np.matmul(invcov0,mean0)
    wi0 = -0.5*np.matmul(np.matmul(mean0.T,invcov0),mean0)[0][0] - 0.5*np.log(np.linalg.det(cov0)) + np.log(0.5)
    big_w1 = -0.5*invcov1
    w1 = np.matmul(invcov1,mean1)
    wi1 = -0.5*np.matmul(np.matmul(mean1.T,invcov1),mean1)[0][0] - 0.5*np.log(np.linalg.det(cov1)) + np.log(0.5)
    # solver
    from sympy import Symbol, solve, Matrix, eye
    def SolverBoundary(x):
        y=Symbol('y')
        pnt = Matrix([x,y])
        equation=pnt.T*(big_w0-big_w1)*pnt+(w0.T-w1.T)*pnt+(wi0-wi1)*eye(1)
        return solve(equation)
    # plot decision boundary
    xs = np.arange(np.min(df['d1']),np.max(df['d1']),0.01)
    ys0 = [SolverBoundary(xs[i])[0][Symbol('y')] for i in range(len(xs))]
    ys1 = [SolverBoundary(xs[i])[1][Symbol('y')] for i in range(len(xs))]

    plt.scatter(df['d1'].values,df['d2'].values,c=df['d3'],cmap='Accent')
    plt.plot(xs,np.array(ys0),color='red', linewidth=2, linestyle='dashed')
    plt.plot(xs,np.array(ys1),color='red', linewidth=2, linestyle='dashed')
    plt.ylim(np.min(df['d2'])-0.2,np.max(df['d2'])+0.2)
    return plt

# Main
# Read Data
train = pd.read_table('D:\\Data\\pattern\\train.txt',delim_whitespace=True,names=['d1','d2','d3'])
test= pd.read_table('D:\\Data\\pattern\\test.txt',delim_whitespace=True,names=['d1','d2','d3'])

#(ii) plotting
train.plot(kind='scatter',x='d1',y='d2',c='d3',colormap='Accent')
test.plot(kind='scatter',x='d1',y='d2',c='d3',colormap='Accent')
train[['d1','d2']].plot(kind='kde')
plt.legend(loc='upper left')
plt.show()
test[['d1','d2']].plot(kind='kde')
plt.legend(loc='upper left')
plt.show()

#(vi)perform 5 training trial
for t in range(5):
    #(iv) sample data
    train_sampled = train.sample(100,random_state=t)
    train_confmat,train_err = ClsError(train_sampled,train_sampled)
    print("###")
    print("Training #"+str(t)+" Confusion Matrix: ",train_confmat[0],train_confmat[1])
    print("Training #"+str(t)+" Error Rate: ",train_err)
    # perform 10 test
    from sklearn.model_selection import KFold
    cv = KFold(10,shuffle=True, random_state=1)
    err_scores = []
    confmats = []
    for i, (_ , test_index) in enumerate(cv.split(test)):
        test_sampled = test.iloc[test_index]
        confmat,err = ClsError(train_sampled,test_sampled)
        err_scores.append(err)
        confmats.append(confmat)
    # get commen set score
    med_idx = err_scores.index(np.percentile(err_scores,50,interpolation='nearest'))
    print("- All Test Scores: ",err_scores)
    print("- Median Test Score(#"+str(med_idx+1)+") Confusion Matrix: ",confmats[med_idx][0],confmats[med_idx][1])
    print("- Median Test Score(#"+str(med_idx+1)+") Error Rate: ",err_scores[med_idx])
    #(vii) decision boundary
    print("Decision Boundary of Training Data")
    plt = DecisionBoundary(train_sampled)
    plt.show()

# Additional Work: classification accuracy corresponding to training data size
training_size = [50,100,150,200,250]
for t in training_size:
    #(iv) sample data
    train_sampled = train.sample(t,random_state=1)
    train_confmat,train_err = ClsError(train_sampled,train_sampled)
    print("###")
    print("Training #"+str(t)+" Confusion Matrix: ",train_confmat[0],train_confmat[1])
    print("Training #"+str(t)+" Error Rate: ",train_err)
    # perform 10 test
    from sklearn.model_selection import KFold
    cv = KFold(10,shuffle=True, random_state=1)
    err_scores = []
    confmats = []
    for i, (_ , test_index) in enumerate(cv.split(test)):
        test_sampled = test.iloc[test_index]
        confmat,err = ClsError(train_sampled,test_sampled)
        err_scores.append(err)
        confmats.append(confmat)
    # get commen set score
    med_idx = err_scores.index(np.percentile(err_scores,50,interpolation='nearest'))
    print("- All Test Scores: ",err_scores)
    print("- Median Test Score(#"+str(med_idx+1)+") Confusion Matrix: ",confmats[med_idx][0],confmats[med_idx][1])
    print("- Median Test Score(#"+str(med_idx+1)+") Error Rate: ",err_scores[med_idx])
    #(vii) decision boundary
    print("Decision Boundary of Training Data")
    plt = DecisionBoundary(train_sampled)
    plt.show()
