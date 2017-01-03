
# coding: utf-8

# # COMP 135 Empirical/Programming Assignment 3

# In[1]:

import sys
import numpy as np
import pandas as pd


# In[2]:

def read_arff(file_name):

    file = open(file_name)
    sample = []
    for line in file:
        if line != "" and (line[0].isdigit() or line[0] == '-'):
            line = line.strip("\n")
            line = line.split(',') 
            sample.append(line)
    for line in sample:
        for i in range(0, len(line)):
            line[i] = float(line[i])
    return sample


# In[3]:

def sig(z):
    return 1.0/(1.0 + np.exp(-z))


# In[4]:

def learn(w,d,traindata,testdata):
    # Set step size for gradient descent
    eta = 0.1
    
    # Read in training data
    train = read_arff(traindata)
    train_data = pd.DataFrame(train).values[:,0:64]
    train_label = pd.DataFrame(train).values[:,[64]]
    
    # Compute the dimension of the input layer and the output layer
    n_input = train_data.shape[1]
    n_output = len(set(train_label.ravel()))
    
    # Translate the training labels into "one hot" vector form
    Id = np.eye(n_output)
    label = []
    for l in train_label:
        label.append(Id[int(l[0])])
    train_label = np.array(label)
    
    # Initialize parameter vectors as a list of matrices
    layer_width = [n_input] + [w] * d + [n_output]
    w_vec = []
    for i in range(len(layer_width)-1):
        w = np.random.uniform(-0.1,0.1,[layer_width[i+1],layer_width[i]])
        w_vec.append(w)
    
    # Training
    for t in range(200):
        for k in range(train_data.shape[0]):
            # Feed Forward
            x = train_data[[k]].T
            s_vec = []
            x_vec = [x]
            for w in w_vec:
                s = w.dot(x)
                x = sig(s)
                s_vec.append(s)
                x_vec.append(x)
            
            # Back propagation:
            y = train_label[[k]].T
            delta_vec = [-(y - x_vec[-1])*x_vec[-1]*(1-x_vec[-1])]
            DELTA_vec = []
            for i in range(1,d+2):
                delta = w_vec[-i].T.dot(delta_vec[-i]) * x_vec[-(i+1)] * (1 - x_vec[-(i+1)])
                DELTA = np.outer(delta_vec[-i],x_vec[-(i+1)])
                delta_vec.insert(0,delta)
                DELTA_vec.insert(0,DELTA)
                w_vec[-i] = w_vec[-i] - eta*DELTA # Update on every example
                
    # Testing
    test = read_arff(testdata)
    test_data = pd.DataFrame(test).values[:,0:64]
    test_label = pd.DataFrame(test).values[:,[64]]
    acc = 0
    for i in range(test_data.shape[0]):
        x = test_data[[i]].T
        # Feed Forward
        for w in w_vec:
            s = w.dot(x)
            x = sig(s)

        if np.argmax(x) == test_label[[i]]:
            acc += 1
    acc = float(acc)/test_data.shape[0]
    return w_vec, acc


# In[5]:

w_vec = [1,2,5,10]
d_vec = [1,2,3,4]

acc_vec = [[],[],[],[],[]]
_, acc = learn(1,0,'optdigits_train.arff','optdigits_test.arff')
acc_vec[0] = [acc]*4

i = 1
for d in d_vec:
    for w in w_vec:
        _, acc = learn(w,d,'optdigits_train.arff','optdigits_test.arff')
        acc_vec[i].append(acc)
    i += 1


# In[6]:

print acc_vec


# ## Discussion:
# 
# From the above plot, we can see how the performance of our neural network (measured by accuracy) vary with different number of layers and different number of layer width. There are a few observations:
# 
# * When d = 0, the width is not relevant, the input layer is directly connected to the output layer, and hence we have a linear classifier. The the performance of the neural network when d = 0 is poor. This is because the structure of the network is too poor to capture the nonlinear structure of the data.
# 
# * When d = 4, the performance of the neural network is also poor.It is because the architecture of the neural network is too rich so that the algorithm does not even converge. It has both low train set error and low test set error.
# 
# * When the depth is moderate, like d = 1,2,3, the neural network peforms well. For a fixed number of hidden layers, the performance of the neural network increases with the width of each layer. The performance is very good when d = 1,2,3 and w = 10 (about 90% accuracy).
# 
# ### Conclusion:
# * It is generally true that, for a fixed number of layers, increase the width of the layers would help increasing performance.
# 
# * It is not true that increasing the depth of the network always help increasing performance.
# 
# * The optimal architecture needs to be determined specifically according to the problems. Cross validations would be useful for doing this.
