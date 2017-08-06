# import tensorflow as tf
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
    string_to_num = {'BusinessTravel':{'non-travel':1,'travely-rarely':2,'frequent':3},
                     'Department':{'ResearchAndDevelopment':1,'Sales':2,'HumanResources':3},
                     'EducationField':{'LifeSciences':1,'Engineering':2},
                     'Gender':{'Male':1,'Female':2},'JobRole':{'manager':1},
                     'MaritalStatus':{'Married':1,'Single':2,'Divorced':3},'Overtime':{'Yes':1,'No':0}}    

    # read the csv file into an array
    df = pd.DataFrame.from_csv('KnowledgeAnalytics_debug.csv',sep=',',index_col=None)
    
    # Deal with attrition column which is existent or non-existent depending
    # on whether we have testing or training data
    if testing == 0:
        # store the attrition column
        attrition = df.get('attrition')
        # drop attrition column
        df = df.drop('attrition',axis=1)
        # turn attrition strings from yes/no into ints
        attrition = attrition.replace(['yes'],1)
        attrition = attrition.replace(['no'],0)
        output_array = attrition

    # manipulate all input data now and turn into array
    # get employee numbers
    id_info = df.get('Employee Number')
              
    # drop employee number,employee count,Over18 since its irrelevant to classification
    drop_list = ['Employee Number','EmployeeCount','Over18']
    df = df.drop(drop_list,axis=1)
    
    # go through dataframe holding csv file and change all strings to numbers
    # using our dictionary mapping
    df = df[(df['BusinessTravel']=='non-travel')|(df['BusinessTravel']=='travely-rarely')
    |(df['BusinessTravel']=='frequent')|(df['Department']=='ResearchAndDevelopment')
    |(df['Department']=='Sales')|(df['Department']=='HumanResources')|
    (df['EducationField']=='LifeSciences')|(df['EducationField']=='Engineering')|
    (df['Gender']=='Male')|(df['Gender']=='Female')|(df['JobRole']=='manager')
    |(df['MaritalStatus']=='Married')|(df['MaritalStatus']=='Single')
    |(df['MaritalStatus']=='Divorced')|(df['Overtime']=='Yes')|(df['Overtime']=='No')]
    df = df.replace(string_to_num)
    
    # turn dataframe into array
    input_array = df.values
    
    if testing == 0:
        return (output_array,id_info,input_array)
    else:
        return(id_info,input_array)
        
      
if __name__ == "__main__":
    (output_array,id_info,input_array) = read_data(0)
    print(input_array)
    print(id_info)
    print(output_array)