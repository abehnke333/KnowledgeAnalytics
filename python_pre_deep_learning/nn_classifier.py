import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold

# knn classifier
# using accuracy as scoring
# algo=auto, nn=20,p=1,weights=distance,test_size=3,accuracy=83.4
# algo=auto, nn=15,p=1,weights=uniform,test_size=3,accuracy=85
# algo=auto, nn=10,p=1,weights=distance,test_size=3,accuracy=84.8
# algo=auto, nn=10,p=1,weights=uniform,test_size=2,accuracy=84.5
# using roc as scoring
# algo=auto,nn=as big as possible,p=1,weights=uniform
def knn_classifier(input_array,output_array,id_info):
    # split our training data into training and validation set
    (x_train,x_val,y_train,y_val) = train_test_split(
            input_array,output_array,test_size=.2,random_state=98)
    
    # create dictionary for grid search nn parameters
    parameters = [{'n_neighbors':[2,5,10,15],'weights':['uniform','distance']
    ,'algorithm':['auto','ball_tree','kd_tree','brute'],'p':[1,2]}]
                  
      
    # use gridSearchCV to train model with different parameters
    clf = GridSearchCV(KNeighborsClassifier(), parameters,scoring='roc_auc')
    clf.fit(x_train, y_train)
    
    # print the best score and best parameters using gridsearch
    print(clf.best_score_)
    print(clf.best_params_)

    y_true, y_pred = y_val, clf.predict(x_val)
    
def knn_out(input_array,output_array,test_x):  
    clf = KNeighborsClassifier(n_neighbors=10,weights='uniform',algorithm='auto',p=1)
    clf.fit(input_array,output_array)
    y_pred = clf.predict(test_x)
    # print test data with associated id_info
    final_output_rf = []
    for i in range(0,len(y_pred)):
        if y_pred[i] == -1:            
            final_output_rf.append('No')
        else:
            final_output_rf.append('Yes') 
    return(final_output_rf)

if __name__ == "__main__":
    input_array = np.loadtxt('train_x.txt')
    output_array = np.loadtxt('train_y.txt')
#    knn_classifier(input_array,output_array,0)
    x_test = np.genfromtxt('test_x.txt')
    output_knn = knn_out(input_array,output_array,x_test)
    np.savetxt('output_knn.csv',output_knn,fmt='%s')