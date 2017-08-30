## basic linear regression tensorflow model
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl

# turn off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define model parameters
learning_rate = .00005
training_epochs = 32000
display_step = 5

# inputs and outputs of neural network
number_of_inputs = 51
number_of_outputs = 1

# Simple 3 layer fully connected model
# define nodes per layer
# attempt 1: 50,100,50: AUC=76, lr=.0001,epoch=30000
# attmpt 2: 25,50,25: AUC = 74, lr=.0001,epoch=30000
# attempt 3: 100,250,100: AUC = 76, lr=.0001,epoch=30000 
layer1_nodes = 50
#layer2_nodes = 100
#layer3_nodes = 50

# Define layers of neural network
# Input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
    

# Layer 1
with tf.variable_scope('layer_1'):
    # need to define as get_variabled because we need these variables to store
    # information as we go through our neural net
    weights = tf.get_variable(name="weights1",shape=[number_of_inputs,layer1_nodes],initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    # need to define as get_variabled because we need these variables to store
    # information as we go through our neural net
    weights = tf.get_variable(name='weights4',shape=[layer1_nodes,number_of_outputs],initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases4', shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_1_output, weights) + biases
    

# Cost function
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None,1))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=Y))
    
# Optimizer
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
# save model
#saver = tf.train.Saver()

   

if __name__ == "__main__":
    # read in input
#    input_array = np.genfromtxt('train_x_non_val.txt') # training data
#    output_array = np.genfromtxt('train_y_non_val.txt').reshape(900,1) # training data
    input_array = np.genfromtxt('train_x.txt') # training data
    output_array = np.genfromtxt('train_y_ce.txt').reshape(1200,1) # training data
    val_x = np.genfromtxt('val_x.txt') # validation set
    val_y = np.genfromtxt('val_y.txt').reshape(300,1) # validation set
    test_x = np.genfromtxt('test_x.txt') # test set

######################## Training #################################################    
    # creaete tensorflow session to run code
    with tf.Session() as session:
        # run global initializer to initialize variables and layers
        session.run(tf.global_variables_initializer())
        # loop optimizer to train network
#        for epoch in range(training_epochs):
        training_cost = 1
        while training_cost > .238:
            # feed in training data and do a step of training
            session.run(optimizer, feed_dict={X: input_array, Y: output_array})
            
#            if epoch % 100 == 0:
#                training_cost = session.run(cost, feed_dict={X: input_array, Y: output_array})
#                print(epoch, training_cost)
            training_cost = session.run(cost, feed_dict={X: input_array, Y: output_array})

        print("Training Is Complete")
        final_cost = session.run(cost, feed_dict={X: input_array, Y: output_array})
        print(final_cost)
        
####################### Validation Testing #########################################
        # repeat for test data as well
        # use model to make prediction using sigmoid function to create confidence
        # that the sample is in the positive class (+1.0)
        output = session.run(prediction, feed_dict={X: val_x}) # run model on test input
        classification_confidence = tf.sigmoid(output) # use sigmoid to get confidence that in +1 class      
#        predicted_class = tf.greater(classification_confidence, 1.2)
#        correct = tf.equal(predicted_class, tf.equal(val_y,1.0))
#        accuracy = tf.reduce_mean( tf.cast(correct, 'float'))
#        print('Accuracy:', accuracy.eval(feed_dict={X: val_x, Y: val_y}))
        
        # probabilities of being class 1
        cc_vals = classification_confidence.eval(feed_dict={X: val_x},session=session)
        
        # create output array for roc curve calculations
        # Create ROC metrics using probabilities as y_score in roc_curve function
        fpr, tpr, thresholds = roc_curve(val_y, cc_vals, pos_label=1)
        roc_auc = auc(fpr, tpr)
        # plot roc curve
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
        # Plot to find optimal cutoff
        # Optimal cutoff: (1 - where the 2 lines intersect)
        i = np.arange(len(tpr)) # index for df
        roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
        roc.ix[(roc.tf-0).abs().argsort()[:1]]       
        # Plot tpr vs 1-fpr
        fig, ax = pl.subplots()
        pl.plot(roc['tpr'])
        pl.plot(roc['1-fpr'], color = 'red')
        pl.xlabel('1-False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic')
        ax.set_xticklabels([])


      
                
      
        
##################### Test Data ###################################################
         
        # run the model on test data
        output = session.run(prediction, feed_dict={X: test_x}) # run model on test input
        classification_confidence_test = tf.sigmoid(output)

        # probabilities of being class 1
        cc_test = classification_confidence_test.eval(feed_dict={X: test_x},session=session)
        
        # use optimal threshold from validation set
        t = .29
        # turn probabilities into classifications and do the same thing
        output_class = []
        for i in range(0,len(cc_test)):
            if cc_test[i] >= t:
                output_class.append('Yes')
            else:
                output_class.append('No')
                
        np.savetxt('tf_output.csv',output_class,fmt='%s')     