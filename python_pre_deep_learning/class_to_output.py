
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

# svm_classify_to_output takes in the data, uses the classifier
# deemed best by svm_classifier, and outputs the final results
# to csv
def svm_classify_to_output(x_train,y_train,id_info,x_test):         
    # use SVC to train model
    clf = SVC(C=1,kernel='linear')
    clf.fit(x_train, y_train)

    # classify test data
    y_pred = clf.predict(x_test)
           
    # print test data with associated id_info
    final_output_svm = []
    for i in range(0,len(y_pred)):
#        final_output_svm[i][0] = id_info[i]
        if y_pred[i] == -1:            
            final_output_svm.append('No')
        else:
            final_output_svm.append('Yes')
        
    return(final_output_svm)


def random_forest_class_to_output(x_train,y_train,id_info,x_test):
    # use gridSearchCV to train model with different parameters
    #1,6,10    
    clf = RandomForestClassifier(max_features='log2',min_samples_leaf=3,
                                 min_samples_split=2,n_estimators=10)
    clf.fit(x_train, y_train)

    # classify test data
    y_pred = clf.predict(x_test)
    
    # print test data with associated id_info
    final_output_rf = []
    for i in range(0,len(y_pred)):
        if y_pred[i] == -1:            
            final_output_rf.append('No')
        else:
            final_output_rf.append('Yes')   
        
    return(final_output_rf)

if __name__ == "__main__":
    id_info = np.zeros((270,1))
    for i in range(0,270):
        id_info[i] = i
    input_array = np.genfromtxt('train_x.txt')
    output_array = np.genfromtxt('train_y.txt')
    x_test = np.genfromtxt('test_x.txt')
    fos = svm_classify_to_output(input_array,output_array,id_info,x_test)
    forf = random_forest_class_to_output(input_array,output_array,id_info,x_test)       
    np.savetxt('fos.csv',fos,fmt='%s')
    np.savetxt('forf.csv',forf,fmt='%s')