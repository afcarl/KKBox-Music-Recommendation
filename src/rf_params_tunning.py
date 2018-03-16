import numpy as np
import pandas as pd
# import lightgbm as lgb
import datetime
import math
import gc
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import xgboost as xgb
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import operator
def roccurve(y,scores):
    fpr=np.zeros(len(y))
    tpr=np.zeros(len(y))
#    fpr[len(y)-1]=1
#    tpr[len(y)-1]=1
    t=sorted(zip(scores,y),reverse=True)
    y=np.array([t[i][1] for i in range(len(t))])
    scores=np.array([t[i][0] for i in range(len(t))])
    for i in range(len(y)):
        if i%100==0:
            print i,'cal'
        temp1=zip(scores[:i+1],y[:i+1])
        temp2=zip(scores[i+1:],y[i+1:])
        TP=sum(np.array([temp1[j][1] for j in range(len(temp1))]))
        FN=sum(np.array([temp2[j][1] for j in range(len(temp2))]))
        FP=len(temp1)-TP
        TN=len(temp2)-FN
        tpr[i]=TP/(sum(np.array([t[j][1] for j in range(len(t))]))).astype(float)
        fpr[i]=FP/(len(t)-sum(np.array([t[j][1] for j in range(len(t))]))).astype(float)
    x=np.append([0],list(fpr))
    y=np.append([0],list(tpr))
    x=list(x)
    y=list(y)
    print 'roc cal end...'
    plt.figure(figsize=(2,2))  
    plt.plot(x,y,"b--",linewidth=1)     
    plt.xlabel("Time(s)") 
    plt.ylabel("Volt")  
    plt.title("Line plot") 
    plt.show() 
    plt.savefig("line.jpg")
    return fpr, tpr

def aucarea(fpr, tpr):
    area=0
    a=[0,0]
    b=[0,0]
    for i in range(len(tpr)):
        b[0]=fpr[i]
        b[1]=tpr[i]
        if b[1]==a[1] and b[0]>a[0]:
            area+=b[1]*(b[0]-a[0])
        a[0]=b[0]
        a[1]=b[1]
    return area

def cal_auc(y_pred_prob,y):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_pred_prob, pos_label=1)
    print sklearn.metrics.auc(fpr, tpr)



print 'loading data...'
concat_experiment=np.load('../inputs/concat4Kaggle.npy')

# np.random.shuffle(concat)

print 'get training data...'
# concat_experiment=concat[:concat.shape[0]/50,:]

concat_experiment_refine=np.concatenate((concat_experiment[:,:8],concat_experiment[:,8+1:]),axis=1) #del original index 8 column
concat_experiment_refine=np.concatenate((concat_experiment_refine[:,:32],concat_experiment_refine[:,32+1:]),axis=1) #del original index 33 col
# concat_experiment_refine=np.concatenate((concat_experiment_refine[:,:5],concat_experiment_refine[:,7:21],concat_experiment_refine[:,23:24],concat_experiment_refine[:,25:]),axis=1)
#22,5,21,6,24

def aveVar(aucs):
    aucs=np.array(aucs)
    average=np.mean(aucs)
    var=np.var(aucs)
    print 'average auc:{}, var :{}'.format(average,var)
    return average
    
    
def modelTrain_RF(model,X_train,y_train,X_test):
    
    model.fit(X_train,y_train)
#     model.fit(X_train,y_train)
    RF_score=model.predict_proba(X_test)
    return RF_score[:,1]


def kFoldCV(model,K,concat_experiment_refine):
    start=0
    dataLen=concat_experiment_refine.shape[0]/K
    aucs=[]
    print '**************** new model RF ***********************'
    for i in range(K):
        X_test_cv=concat_experiment_refine[start:(start+dataLen),:-1]
        y_test_cv=concat_experiment_refine[start:(start+dataLen),-1]
        y_test_cv=y_test_cv.reshape([y_test_cv.shape[0],])

        train_idx=np.array(range(0,start)+range(start+dataLen,concat_experiment_refine.shape[0]))
        X_train_cv=concat_experiment_refine[train_idx[:None],:-1]
        y_train_cv=concat_experiment_refine[train_idx[:,None],-1]
        y_train_cv=y_train_cv.reshape([y_train_cv.shape[0],])
        print 'X_train shape:{}, y_train_cv:{},X_test shape:{}, y_test shape:{}'.format(X_train_cv.shape,y_train_cv.shape,X_test_cv.shape,y_test_cv.shape)
        start+=dataLen

        y_pred=modelTrain_RF(model,X_train_cv,y_train_cv,X_test_cv)

#         model.fit(X_train_cv,y_train_cv,eval_metric='auc')
#         y_pred = model.predict(X_test_cv)
        auc=roc_auc_score(y_test_cv, y_pred)
#         accuracy=accuracy_score(y_test_cv, y_pred)
#         print "auccuracy is",accuracy
#         aucs.append(auc)
        print "auc is",auc
        aucs.append(auc)
    print aucs
    return aveVar(aucs)



n_estimators=[10,20,30,50,70]
max_depths=[5,10,20,30,50,70]
min_sample_leafs=[1,10,30,50,70]
tunning_results={}
print 'tunning params for RF...'
for n_estimator in n_estimators:
    for max_depth in max_depths:
        for min_sample_leaf in min_sample_leafs:
            print 'n_estimator:{},max_depth:{},min_sample_leaf:{}'.format(n_estimator,max_depth,min_sample_leaf)
            RF=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimator,min_samples_leaf=min_sample_leaf, random_state=0)
            ave_auc=kFoldCV(RF,5,concat_experiment_refine)
            tunning_results['{},{},{}'.format(n_estimator,max_depth,min_sample_leaf)]=ave_auc


print sorted(tunning_results.iteritems(), key=lambda (k,v): (v,k),reverse=True)
print 'dumping..'
pkl.dump(tunning_results,open('./tunning_rf.pkl','w'))
print 'done'

