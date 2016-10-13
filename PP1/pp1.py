import sys
import numpy as np

# Task 3.1 Evaluating Decision Trees
# Run weka and evaluate J48 on the test sets,
# and the results of accuracies are as following:

J48_ionosphere = 0.91453
J48_irrelevant = 0.645
J48_mfeat_fourier = 0.746627
J48_spambase = 0.915906

print "***************************************************"
print "***                                             ***"
print "***                   Task 3.1                  ***"
print "***                                             ***"
print "***************************************************"
print "J48_ionosphere = ", J48_ionosphere
print "J48_irrelevant = ", J48_irrelevant
print "J48_mfeat_fourier = ", J48_mfeat_fourier
print "J48_spambase = ", J48_spambase

# Task 3.2 Reading data files

def read_arff(file_name):
	
	# This function read in arff file from an url

    file = open(file_name)
    sample = []
    for line in file:
        if line != "" and line[0].isdigit():
            line = line.strip("\n")
            line = line.split(',') 
            sample.append(line)
    for line in sample:
        for i in range(0, len(line)-1):
            line[i] = float(line[i])
    return sample

train_data_1 = read_arff("ionosphere_train.arff")
train_data_2 = read_arff("irrelevant_train.arff")
train_data_3 = read_arff("mfeat-fourier_train.arff")
train_data_4 = read_arff("spambase_train.arff")

test_data_1 = read_arff("ionosphere_test.arff")
test_data_2 = read_arff("irrelevant_test.arff")
test_data_3 = read_arff("mfeat-fourier_test.arff")
test_data_4 = read_arff("spambase_test.arff")

# Task 3.3 Implementing kNN

def split_label(data):

	# This function split the data into data points and one column of labels

    example = []
    label = []
    for line in data:
        example.append(line[0:len(line)-1])
        label.append(line[len(line)-1])
    return np.array(example), np.array(label)

def kNN(test_data,train_data,k):
    
	# This function performs the kNN and compute the accuracy on test data.

    train_points, train_labels = split_label(train_data)
    test_points, test_labels = split_label(test_data)
    
    test_labels_hat = []
    for point in test_points:
        distances = np.sqrt(np.sum((point - train_points)**2,1))
        ind = np.argsort(distances)[:k]
        
        vote = dict.fromkeys(set(train_labels),0) # Create a dictionary with all the labels
        for indice in ind:
            vote[train_labels[indice]] += 1
        test_labels_hat.append(max(vote, key = vote.get)) # predict the label for test point by a majority vote
        
    accuracy = (sum(np.array(test_labels_hat) == np.array(test_labels)) + 0.0) / len(test_labels)
    return accuracy

print "***************************************************"
print "***                                             ***"
print "***                   Task 3.4                  ***"
print "***                                             ***"
print "***************************************************"

accuracy_vec = []
for k in range(1,26): # for k from 1 to 25, evaluate kNN and record the accuracy in a vector
    accuracy = kNN(test_data_1, train_data_1, k)
    accuracy_vec.append(accuracy)
print "Accuracy on ionosphere:"
print accuracy_vec

accuracy_vec = []
for k in range(1,26):
    accuracy = kNN(test_data_2, train_data_2, k)
    accuracy_vec.append(accuracy)
print "Accuracy on irrelevant:"
print accuracy_vec

accuracy_vec = []
for k in range(1,26):
    accuracy = kNN(test_data_3, train_data_3, k)
    accuracy_vec.append(accuracy)
print "Accuracy on mfeat-fourier:"
print accuracy_vec

accuracy_vec = []
for k in range(1,26):
    accuracy = kNN(test_data_4, train_data_4, k)
    accuracy_vec.append(accuracy)
print "Accuracy on spambase:"
print accuracy_vec

print "***************************************************"
print "***                                             ***"
print "***                   Task 3.5                  ***"
print "***                                             ***"
print "***************************************************"

def compute_gain(train_data,i):
    
	# This function compute the SplitGain for the ith feature

    feature = []
    train_points, train_labels = split_label(train_data)
    labels = set(train_labels)
    N = len(train_data)
    
    ind  = np.argsort(train_points[:,i]) # indice that sort the ith column of the data
    
    count = []
    gain = 0
    for l in labels:
        count.append(list(train_labels).count(l)) # count the number of appearances for each label
    for p in count: # compute the gain
        pk = (p + 0.0)/sum(count)
        if pk != 1 and pk != 0:
            gain += - pk * np.log2(pk)
    
    for k in range(0,5):
        ind_k = ind[k*N/5:(k+1)*N/5]
        count_k = []
        gain_k = 0
        for l in labels:
            count_k.append(list(train_labels[ind_k]).count(l))
        for p in count:
            pk = (p + 0.0)/sum(count_k)
            if pk != 1 and pk != 0:
                gain_k += - pk * np.log2(pk)
        gain += - ((len(ind_k)+0.0)/N) * gain_k
    return gain

def choose_feature(train_data,test_data,n):

	# This function evalutate kNN with the top n selected features
    
    split_gain = []
    for i in range(0,len(train_data[0])-1):
        split_gain.append(compute_gain(train_data,i))
    ind = np.argsort(split_gain)[-n:]
    
    train_points, train_labels = split_label(train_data)
    test_points, test_labels = split_label(test_data)
    train_points = train_points[:,ind]
    test_points = test_points[:,ind]
    
    test_labels_hat = []
    for point in test_points:
        distances = np.sqrt(np.sum((point - train_points)**2,1))
        ind = np.argsort(distances)[:5]
        
        vote = dict.fromkeys(set(train_labels),0)
        for indice in ind:
            vote[train_labels[indice]] += 1 
        test_labels_hat.append(max(vote, key = vote.get))
        
    accuracy = (sum(np.array(test_labels_hat) == np.array(test_labels)) + 0.0) / len(test_labels)
    return accuracy

accuracy_vec = []
n_vec = np.arange(1,len(train_data_1[0])-1)
for n in n_vec:
    accuracy = choose_feature(train_data_1, test_data_1, n)
    accuracy_vec.append(accuracy)
print "Accuracy on ionosphere:"
print accuracy_vec

accuracy_vec = []
n_vec = np.arange(1,len(train_data_2[0])-1)
for n in n_vec:
    accuracy = choose_feature(train_data_2, test_data_2, n)
    accuracy_vec.append(accuracy)
print "Accuracy on irrelevant:"
print accuracy_vec

accuracy_vec = []
n_vec = np.arange(1,len(train_data_1[0])-1)
for n in n_vec:
    accuracy = choose_feature(train_data_3, test_data_3, n)
    accuracy_vec.append(accuracy)
print "Accuracy on mfeat-fourier:"
print accuracy_vec

accuracy_vec = []
n_vec = np.arange(1,len(train_data_1[0])-1)
for n in n_vec:
    accuracy = choose_feature(train_data_4, test_data_4, n)
    accuracy_vec.append(accuracy)
print "Accuracy on spambase:"
print accuracy_vec

