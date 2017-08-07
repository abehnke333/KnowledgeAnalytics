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
    final_output_svm = np.zeros((270,2),dtype=int)
    for i in range(0,len(y_pred)):
        final_output_svm[i][0] = id_info[i]
        final_output_svm[i][1] = y_pred[i]
        
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
    final_output_rf = np.zeros((270,2),dtype=int)
    for i in range(0,len(y_pred)):
        final_output_rf[i][0] = id_info[i]
        final_output_rf[i][1] = y_pred[i]    
        
    return(final_output_rf)

if __name__ == "__main__":
    input_array = np.loadtxt('train_x.txt')
    output_array = np.loadtxt('train_y.txt')
    id_info = np.loadtxt('id_info.txt')
    x_test = np.loadtxt('test_x.txt')
    fos = svm_classify_to_output(input_array,output_array,id_info,x_test)
    forf = random_forest_class_to_output(input_array,output_array,id_info,x_test)
#    for i in range(0,len(fos)):
#        print(fos[i][1],forf[i][1])
    # write output to csv
    d = {'Attrition':forf[:][:]}
    out_a = pd.DataFrame(data=d,index=forf[:][:])
    print(out_a)
    mapping = {-1:'No',1:'Yes'}
    out_a = out_a.replace(mapping)
    print(out_a)          
    np.savetxt('forf.csv',out_a,delimiter=',')
    
    