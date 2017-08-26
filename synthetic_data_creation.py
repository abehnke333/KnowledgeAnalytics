# import tensorflow as tf
import numpy as np
import pandas as pd
import scipy as sc
import sklearn as sk
from sklearn import preprocessing
import statistics as st
import Data

# add_synthetic_data takes in the data (n samples of m features) and slightly
# changes the data as to add more samples to the data. This is similar to 
# shifting pixels in images by a set amount to get more samples that will increase
# the robustness of the model
def add_synthetic_data(df):

    # We don't want to change the categorical columns so we will leave them
    # unaffected in the df. Thus, let's create a list to make sure they
    # are only copied and not changed
    categorical_input = ['BusinessTravel','Department','EducationField','Gender','JobRole',
            'MaritalStatus','OverTime']
    
    # create dataframe to hold needed information
    col_names = df.head()
    df_stats = pd.DataFrame(data=col_names,index=None)
    
    # get statistics of every non-catagorical field 
    # drop categorical fields      
    df_stats = df_stats.drop(categorical_input,axis=1)
    
    
    # go through main dataframe and get 
    # mean,median,mode,stdev,stdvar for each field
    for column in df:
        df_stats[column][0] = 
    
    
    
    

    
    

    return(s_df)
        
    
if __name__ == "__main__":
    df = pd.DataFrame.from_csv('AnalyticsChallenge1-Train.csv',sep=',',index_col=None)
    s = add_synthetic_data(df)