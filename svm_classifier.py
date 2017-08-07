import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# SVM Classifier script used to find the best parameters for the 
# classification at hand
# Both predictions used a 20% of training data validation set
# best svm: C = 1e-6,linear kernel at 84.1% accuracy
# best rfc: Max_features = sqrt, Min_samples_leaf = 2, 
#   min_samples_split = 6, n_estimators = 10 at 84.58 accuracy
def svm_classifier(input_array,output_array,id_info):
    # split our training data into training and validation set
    (x_train,x_val,y_train,y_val) = train_test_split(
            input_array,output_array,test_size=.2,random_state=0)
    
    # create dictionary for grid search cv parameters
    # param1: linear kernel, gamma
    # param2: poly kernel, degree 2->6,
    # param3: rbf kernel
    parameters = [{'kernel':['linear'],'C':[1e-6,1e-3,1]},
                   {'kernel':['rbf'],'C':[1e-6,1e-3,1]},
                   {'kernel':['poly'],'degree':[2,3,4,5,9],'gamma':[1e-3,1]
                   ,'C':[1e-6,1e-3,1,100]}]

      
    # use gridSearchCV to train model with different parameters
    clf = GridSearchCV(SVC(), parameters, cv=8)
    clf.fit(x_train, y_train)
    
    # print the best score and best parameters using gridsearch
    print(clf.best_score_)
    print(clf.best_params_)

    y_true, y_pred = y_val, clf.predict(x_val)
    
    # create own accuracy guage
    diff = abs(y_true-y_pred)
    s = sum(diff)/2
    accuracy = 1-(s/len(y_true))
    print('accuracy: ', accuracy)    


def random_forest_class(input_array,output_array,id_info):
    # split our training data into training and validation set
    (x_train,x_val,y_train,y_val) = train_test_split(
            input_array,output_array,test_size=.2,random_state=0)
    
    parameters = [{'n_estimators':[10,20,30],'max_features':['sqrt','log2'],
                   'min_samples_split':[2,4,6],'min_samples_leaf':[1,2,3]}]

    # use gridSearchCV to train model with different parameters
    clf = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
    clf.fit(x_train, y_train)
    
    # print the best score and best parameters using gridsearch
    print(clf.best_score_)
    print(clf.best_params_)

    y_true, y_pred = y_val, clf.predict(x_val)
    
    # create own accuracy guage
    diff = abs(y_true-y_pred)
    s = sum(diff)/2
    accuracy = 1-(s/len(y_true))
    print('accuracy: ', accuracy)    


if __name__ == "__main__":
    input_array = np.loadtxt('train_x.txt')
    output_array = np.loadtxt('train_y.txt')
    svm_classifier(input_array,output_array,0)
    random_forest_class(input_array,output_array,0)