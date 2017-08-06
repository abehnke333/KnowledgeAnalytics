import tensorflow as tf
import csv
import numpy as np
import pandas as pd

# reads in training and testing data
# Input: testing = int. If testing is 1 then don't read in output (col 2)
# else read in the input and output data
# Output: if testing = 0, a 2D input array and 1D output array. If testing = 1
# then just a 2D input array
def read_data(testing):    
    # create hashmap to simply turn strings into numbers needed for analysis
    # key: string
    # value: number represented input string
    string_to_num = {'yes':1,'no':0,'non-travel':1,'travely-rarely':2,'frequent':3,
                     'ResearchAndDevelopment':1,'Sales':2,'HumanResources':3,'LifeSciences':1,
                     'Engineering':2,'Male':1,'Female':2,'Manager':1,'Married':1,'Single':2,
                     'Divorced':3,'manager':1,'Yes':1,'No':0}            

    # read the csv file into an array
    df = pd.DataFrame.from_csv('KnowledgeAnalytics_debug.csv',sep=',',index_col=None)
    array = df.values

    # get size of array
    (n,d) = np.shape(array)
    
    # if testing = 0, then we need to return input and output array
    # else just return an input_array
    if testing == 0:
        # create input and output arrays
        input_array = np.zeros((n,d-1)) # take out attrition column
        output_array = np.zeros((n,1))
        
        # go through array, convert all strings to ints and put into input/output arrays
        for i in range(0,n):
            for j in range(0,d-1):
                if j == 0:
                    input_array[i][j] = array[i][j]
                elif j == 1:
                    output_array[i] = string_to_num[array[i][j]]
                else:
                    if array[i][j] in string_to_num:
                        input_array[i][j-1] = string_to_num[array[i][j]]
                    else:
                        input_array[i][j-1] = array[i][j]                      
        return(input_array,output_array)
    
    else:
        input_array = np.zeros(n,d)
        for i in range(0,n):
            for j in range(0,d):                
                if array[i][j] in string_to_num:
                    input_array[i][j] = string_to_num[array[i][j]]
                else:
                    input_array[i][j] = array[i][j]       
        return(input_array)
                    
      
if __name__ == "__main__":
    (input_array,output_array) = read_data(0)
    print(input_array)
    print(output_array)