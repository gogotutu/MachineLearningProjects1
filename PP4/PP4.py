
# coding: utf-8

# # COMP 135 Empirical/Programming Assignment 4

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
    return np.array(sample)


# In[3]:

def PwM_Primal(file):
    
    train_filename = file + 'Train.arff'
    test_filename = file + 'Test.arff'
    train_file = read_arff(train_filename)
    test_file = read_arff(test_filename)
    
    N = train_file.shape[0]
    M = train_file.shape[1]
    N_test = test_file.shape[0]

    train_data = train_file[:,0:M-1]
    test_data = test_file[:,0:M-1]
    train_data = np.concatenate((np.ones([N,1]),train_data), axis = 1)
    test_data = np.concatenate((np.ones([N_test,1]),test_data), axis = 1)
    train_label = train_file[:,[-1]]
    test_label = test_file[:,[-1]]

    A = 0
    for i in range(N):
        A += np.linalg.norm(train_data[i])
    A = A/N

    pi = 0.1 * A
    I = 50
    
    w = np.zeros([1,M])
    for k in range(I):
        for i in range(N):
            O = np.sign(w.dot(train_data[[i]].T))
            if train_label[[i]] * w.dot(train_data[[i]].T) < pi:
                w += train_label[[i]] * train_data[[i]]

    acc = 0
    for i in range(N_test):
        if np.sign(w.dot(test_data[[i]].T)) == test_label[[i]]:
            acc += 1
    acc = float(acc)/N_test
    return acc


# In[4]:

def K_POLY(x_k,x_i,d):
    return (x_k.dot(x_i.T) + 1)**d


# In[5]:

def K_RBF(x_k,x_i,s):
    return np.exp(-np.linalg.norm(x_k - x_i)**2 / (2*s**2))


# In[6]:

def PwM_POLY(file,d):

    train_filename = file + 'Train.arff'
    test_filename = file + 'Test.arff'

    train_file = read_arff(train_filename)
    test_file = read_arff(test_filename)
    N = train_file.shape[0]
    M = train_file.shape[1] - 1
    N_test = test_file.shape[0]

    train_data = train_file[:,0:M]
    test_data = test_file[:,0:M]
    train_label = train_file[:,[-1]]
    test_label = test_file[:,[-1]]
    x = train_data
    y = train_label

    A = 0
    for i in range(N):
        A += K_POLY(x[[i]],x[[i]],d)**(1.0/2)
    A = A/N

    pi = 0.1 * A
    I = 50

    alpha = np.zeros([N,1])
    w = np.zeros([1,M])
    K = np.zeros([N,N])
    for k in range(N):
        for i in range(N):
            K[k,i] = K_POLY(x[[k]],x[[i]],d)

    for iter in range(I):
        for i in range(N):
            S = 0
            for k in range(N):
                S += alpha[[k]] * y[[k]] * K[k,i]
            O = np.sign(S)
            if y[[i]] * S < pi:
                alpha[[i]] += 1

    acc = 0
    for i in range(N_test):
        S = 0
        for k in range(N):
            S += alpha[[k]] * y[[k]] * K_POLY(x[[k]],test_data[[i]],d)
        O = np.sign(S)
        if O == test_label[[i]]:
            acc += 1
    acc = float(acc)/N_test
    return acc


# In[7]:

def PwM_RBF(file,s):

    train_filename = file + 'Train.arff'
    test_filename = file + 'Test.arff'

    train_file = read_arff(train_filename)
    test_file = read_arff(test_filename)
    N = train_file.shape[0]
    M = train_file.shape[1] - 1
    N_test = test_file.shape[0]

    train_data = train_file[:,0:M]
    test_data = test_file[:,0:M]
    train_label = train_file[:,[-1]]
    test_label = test_file[:,[-1]]
    x = train_data
    y = train_label

    A = 0
    for i in range(N):
        A += K_RBF(x[[i]],x[[i]],s)**(1.0/2)
    A = A/N

    pi = 0.1 * A
    I = 50

    alpha = np.zeros([N,1])
    w = np.zeros([1,M])
    K = np.zeros([N,N])
    for k in range(N):
        for i in range(N):
            K[k,i] = K_RBF(x[[k]],x[[i]],s)

    for iter in range(I):
        for i in range(N):
            S = 0
            for k in range(N):
                S += alpha[[k]] * y[[k]] * K[k,i]
            O = np.sign(S)
            if y[[i]] * S < pi:
                alpha[[i]] += 1

    acc = 0
    for i in range(N_test):
        S = 0
        for k in range(N):
            S += alpha[[k]] * y[[k]] * K_RBF(x[[k]],test_data[[i]],s)
        O = np.sign(S)
        if O == test_label[[i]]:
            acc += 1
    acc = float(acc)/N_test
    return acc


# In[8]:

s_vec = [0.1, 0.5, 1, 2, 5, 10]
d_vec = [1, 2, 3, 4, 5]

def print_performance(filename):
    acc_primal = PwM_Primal(filename)
    acc_POLY = []
    for i in range(5):
        acc_POLY.append(PwM_POLY(filename,d_vec[i]))
    acc_RBF = []
    for i in range(6):
        acc_RBF.append(PwM_RBF(filename,s_vec[i]))
    print "Primal: ", acc_primal
    print "Polynomial: ", acc_POLY
    print "RBF: ", acc_RBF 


# In[9]:

print_performance("A")


# ### Discussion:
# 
# In this dataset, for small values of the parameters, the RBF kernel works better than Polynomial kernel. For large values of the parameters the contrary happens. This suggests the separation hyperplane is curly for this dataset.

# In[10]:

print_performance("B")


# ### Discussion:
# 
# In this dataset, both kernels follow the same pattern that the accuracy firstly increases and then decreases.

# In[11]:

print_performance("C")


# ### Discussion:
# 
# In this dataset, Polynomial kernels follow the same pattern as in previous one. The RBF kernel in genreal follows the same pattern, with a little exception that the s = 10 case is a little better than s = 5 case. However, the optimal s is obtained at s = 2.

# In[12]:

print_performance("back")


# ### Discussion:
# 
# In this dataset, Polynomial kernel works better than RBF kernel for large parameter values, and RBF kernel works better than Polynomial kernel for small parameter values. This again suggests that separation hyperplane is probably curly.

# In[13]:

print_performance("breast")


# ### Discussion:
# 
# For this dataset, it is not very sensitive to the parameters. Notice that they all get very high accuracy and does not very much.

# In[14]:

print_performance("sonar")


# ### Discussion:
# Both kernels follow the same pattern. 

# ### Summary:
# 
# * The Primal perceptron and the Polynomial kernel percetron with d = 1 are identical to each other. These are plotted as the first and the second point on the blue curve. This is because when d = 1, the Polynomial kernel percetron is mathematically equivalent to the Primal perceptron with a column of 1 added to the feature.
# 
# * It is generally true, for all datasets and both kernels, that the optimal value for the parameter is chosen for some medium value (neither the largest nor the smallest value). Too large or too small values for the algorithm would result in overfitting or underfitting problem. That's why the optimal would be reached by some medium value.
