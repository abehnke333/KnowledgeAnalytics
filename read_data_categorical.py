# import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import scipy as sc
import sklearn as sk
import category_encoders as ce
from sklearn import preprocessing
import synthetic_data_creation

# reads in training and testing data
# Input: testing = int. If testing is 1 then don't read in output (col 2)
# else read in the input and output data
# Output: if testing = 0, a 2D input array and 1D output array. If testing = 1
# then just a 2D input array
def read_data(testing):       
    # read the csv file into an array
    df = pd.DataFrame.from_csv('AnalyticsChallenge1-Train.csv',sep=',',index_col=None)
    
    # Deal with attrition column which is existent or non-existent depending
    # on whether we have testing or training data
    if testing == 0:
        # add_synthetic_data takes in the csv in data frame format and perturbers
        # each sample by a very small ammount to create synthetic data to train
        # the neural network since we need a lot of data. This will take us from 
        # 1200 test cases to at least ~3600 test cases since we can perturbe positively
        # and negatively. Technically, we can change variation size and create a lot
        # more test cases while still keeping true with the pattern recognition
        df = add_synthetic_data(df)
        
        # store the attrition column
        attrition = df.get('Attrition')
        # drop attrition column
        df = df.drop('Attrition',axis=1)
        # turn attrition strings from yes/no into ints
        attrition = attrition.replace(['Yes'],1)
        attrition = attrition.replace(['No'],-1)
        output_array = attrition

    # manipulate all input data now and turn into array
    # get employee numbers
    id_info = df.get('EmployeeNumber')
              
    # drop employee number,employee count,Over18 since its irrelevant to classification
    drop_list = ['EmployeeNumber','EmployeeCount','Over18','StandardHours']
    df = df.drop(drop_list,axis=1)
    
    # create a new dataframe so we can do our categorical encoding
    new_df = df.select_dtypes(include=['object']).copy()
    
    # specify the columns to fit using backward difference encoding
    encode_list = ['BusinessTravel','Department','EducationField','Gender','JobRole',
            'MaritalStatus','OverTime']
    encoder = ce.OneHotEncoder(cols=encode_list)
    encoder.fit(new_df)
    
    # transform the data 
    new_df = encoder.transform(new_df)
    # drop -1 cols
    new_df = new_df.drop(['BusinessTravel_-1','Department_-1','EducationField_-1',
                         'Gender_-1','JobRole_-1','MaritalStatus_-1','OverTime_-1'],axis=1)
    
    # drop unencoded categorical columns
    df = df.drop(encode_list,axis=1)
    
    # join df_norm and hotone encoded dataframe
    df_norm = df.join(new_df)
    
    # normalize df columns then join with encoded
    df_final = sk.preprocessing.normalize(df_norm,norm='l2',axis=0)
    
#    # normalize all data
#    df_final = sk.preprocessing.normalize(df_final,norm='l2')
    input_array = df_final

    
    if testing == 0:
        return (output_array,input_array)
    else:
        return(id_info,input_array)
        
      
if __name__ == "__main__":
    (output_array,input_array) = read_data(0)
    np.savetxt('test_x_synth.txt',input_array,fmt='%f')
    np.savetxt('train_y_synth.txt',output_array,fmt='%f')
#    np.savetxt('id_info.txt',id_info,fmt='%d')
    
