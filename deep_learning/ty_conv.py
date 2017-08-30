import numpy as np
from sklearn.model_selection import train_test_split

#old_y = np.genfromtxt('train_y.txt').reshape(1200,1)
#new_y = np.zeros((1200,1))
#for i in range(0,len(old_y)):
#    if old_y[i] == -1:
#        new_y[i] = 0.0
#    else:
#        new_y[i] = 1.0
#        
#np.savetxt('train_y_ce.txt',new_y,fmt='%s')    

x = np.genfromtxt('train_x.txt')
y = np.genfromtxt('train_y_ce.txt').reshape(1200,1)
train_x_non_val_2,val_x_2,train_y_non_val_2,val_y_2 = train_test_split(x,y,test_size=.25,random_state=9) 
        


#print(train_x_non_val)
#print(train_y_non_val)
#print(val_x)
#print(val_y) 
np.savetxt('train_x_non_val_5.txt',train_x_non_val_2,fmt='%s')
np.savetxt('train_y_non_val_5.txt',train_y_non_val_2,fmt='%s')
np.savetxt('val_x_5.txt',val_x_2,fmt='%s')
np.savetxt('val_y_5.txt',val_y_2,fmt='%s')