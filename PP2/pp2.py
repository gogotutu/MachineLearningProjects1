import sys
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# This function reads a file for a given folder with its index,
# and return a list of strings of all the words in the document

def read_file(folder,index):
    dir = folder + '/' + str(index) + '.clean'
    file = open(dir,'r')
    text = file.read().split()
    return text


# This function reads the label files (index_train and index_test)
# for a given folder and a given size (0.1 to 1).
# It returns a dictionary with document index as keys and "yes" or "no" as values,
# It also returns two other dictionaries for separated positive and negative examples.

def read_label(folder, file_name, size):
    dir = folder + '/' + file_name
    file = open(dir,'r')
    l = file.read().split("\n")
    if "" in l:
        l.remove("")
    
    N = int(size * len(l))
    label = {}
    label_pos = {}
    label_neg = {}
    for line in l[0:N]:
        [key, val] = line.split("|")[0:2]
        label[int(key)] = val
        if val == "yes":
            label_pos[int(key)] = val
        else:
            label_neg[int(key)] = val
    
    return label, label_pos, label_neg




# This function trains our model with two types of variant
# It takse in four arguments:
# 
# folder: "ibmmac" or "sport" for two different folders
# size: 0.1, 0.2, ..., 1 for different training set sizes
# m: is the smoothing parameter
# train_type: 1 or 2 for two types of variant
# 
# It returns four variables:
# pos_rate: P(+)
# neg_rate: P(-)
# vocab_pos: is a dictionary with all words in vocabulary as its keys and P(word|+) as its values
# vocab_neg: is a dictionary with all words in vocabulary as its keys and P(word|-) as its values

def train(folder, size, m, train_type):
    
    # Call previous defined function to import the index and labels 
    train_labels, label_pos, label_neg = read_label(folder, "index_train", size)
    
    # Create a vocabulary using documents in the training set
    words = []
    for i in train_labels.keys():
        words += read_file(folder,i)
    all_words = set(words)
    
    # Initialization
    vocab_pos = dict.fromkeys(all_words,0)
    vocab_neg = dict.fromkeys(all_words,0)
    
    if train_type == 1: # If type 1 variant
        for i in label_pos.keys(): # For all documents in positive examples
            text = read_file(folder,i) # Read the document
            for word in text: # For all words in the document
                vocab_pos[word] += 1 # Count for each appearance of the word
        for i in label_neg.keys(): # Simlar for all the negative examples
            text = read_file(folder,i)
            for word in text:
                vocab_neg[word] += 1
        
        # Define #c and V for the type 1 case
        num_c_pos = sum(vocab_pos.values())
        num_c_neg = sum(vocab_neg.values())
        V = len(all_words)
        
        # Compute P(word|+) and P(word|-)
        for key in vocab_pos.keys():
            vocab_pos[key] = float(vocab_pos[key] + m) / (num_c_pos + m * V)
        for key in vocab_neg.keys():
            vocab_neg[key] = float(vocab_neg[key] + m) / (num_c_neg + m * V)
    
    # Similar for type 2 variant
    if train_type == 2:
        for i in label_pos.keys():
            text = read_file(folder,i)
            for word in set(text):
                vocab_pos[word] += 1    
        for i in label_neg.keys():
            text = read_file(folder,i)
            for word in set(text):
                vocab_neg[word] += 1
        
        # Different definitions for #c and V for type 2 variant
        num_c_pos = len(label_pos)
        num_c_neg = len(label_neg)
        V = 2
        
        for key in vocab_pos.keys():
            vocab_pos[key] = float(vocab_pos[key] + m) / (num_c_pos + m * V)
        for key in vocab_neg.keys():
            vocab_neg[key] = float(vocab_neg[key] + m) / (num_c_neg + m * V)        

    # Compute P(+) and P(-)
    pos_rate = float(len(label_pos))/len(train_labels)
    neg_rate = float(len(label_neg))/len(train_labels)
    
    return vocab_pos, vocab_neg, pos_rate, neg_rate





# This function makes the prediction on the test data
# It takes the same four arguments as in function "train"
# and returns the predicted labels for the test data.
# It also returns the test labels for convenience

def predict(folder, size, m, train_type):
    
    # Read in test labels
    test_label, test_pos, test_neg = read_label(folder, "index_test", 1)
    
    # Train our model
    vocab_pos, vocab_neg, pos_rate, neg_rate = train(folder, size, m, train_type)
    
    # Initialize the predicted labels
    predicted_labels = dict.fromkeys(test_label.keys(),"")
    
    if train_type == 1:
        for i in test_label.keys(): # For all documents in the test set
            text = read_file(folder,i) # Read in a document
            
            # Initialize + and - score to be P(+) and P(-)
            score_pos = np.log(pos_rate)
            score_neg = np.log(neg_rate)
            for word in text: # For all words in the document
                if (word in vocab_pos) or (word in vocab_neg): # If the word appears in either class, update the scores
                    score_pos += np.log(vocab_pos[word])
                    score_neg += np.log(vocab_neg[word])
            # The case when m = 0 is taken care of by the numpy.log function, as shown below
            
            # Predict by comparing the two scores
            if score_pos > score_neg:
                predicted_labels[i] = "yes"
            else:
                predicted_labels[i] = "no"
    
    if train_type == 2: # Type 2 is a little different
        for i in test_label.keys():
            text = read_file(folder,i)
            
            score_pos = math.log(pos_rate)
            score_neg = math.log(neg_rate)
            for word in vocab_pos.keys(): # For all words in vocabulary
                if word in text: # If the word is in the document, update the scores by P(1|+/-)
                    score_pos += np.log(vocab_pos[word])
                    score_neg += np.log(vocab_neg[word])
                else: # If the word is NOT in the document, update the score by P(0|+/-) = 1 - P(1|+/-)
                    score_pos += np.log(1 - vocab_pos[word])
                    score_neg += np.log(1 - vocab_neg[word])
            # The case when m = 0 is taken care of by the numpy.log function, as shown below
            
            # Predict by comparing the two scores
            if score_pos > score_neg:
                predicted_labels[i] = "yes"
            else:
                predicted_labels[i] = "no"
    
    return test_label, predicted_labels


# This function compute the accuracy of our algorithm
# Accuracy = correctly predicted labels / all labels

def compute_accuracy(test_labels, predicted_labels):
    acc = 0
    for key in test_labels.keys():
        if test_labels[key] == predicted_labels[key]:
            acc += 1
    acc = float(acc)/len(test_labels)
    return acc








################################################################################
#####																	   #####
#####										   							   #####
#####								EVALUATION                             #####
#####										   							   #####
#####								  PART I							   #####
#####																	   #####
################################################################################

learning_curve_ibmmac_221 = []
learning_curve_ibmmac_222 = []
learning_curve_ibmmac_223 = []
learning_curve_ibmmac_224 = []
size_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "ibmmac", size = s, m = 0, train_type = 1)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_ibmmac_221.append(acc)

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "ibmmac", size = s, m = 1, train_type = 1)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_ibmmac_222.append(acc)

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "ibmmac", size = s, m = 0, train_type = 2)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_ibmmac_223.append(acc)

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "ibmmac", size = s, m = 1, train_type = 2)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_ibmmac_224.append(acc)

print learning_curve_ibmmac_221
print learning_curve_ibmmac_222
print learning_curve_ibmmac_223
print learning_curve_ibmmac_224




learning_curve_sport_221 = []
learning_curve_sport_222 = []
learning_curve_sport_223 = []
learning_curve_sport_224 = []
size_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "sport", size = s, m = 0, train_type = 1)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_sport_221.append(acc)

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "sport", size = s, m = 1, train_type = 1)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_sport_222.append(acc)

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "sport", size = s, m = 0, train_type = 2)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_sport_223.append(acc)

for s in size_vec:
    test_labels, predicted_labels = predict(folder = "sport", size = s, m = 1, train_type = 2)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_sport_224.append(acc)

print learning_curve_sport_221
print learning_curve_sport_222
print learning_curve_sport_223
print learning_curve_sport_224



################################################################################
#####																	   #####
#####										   							   #####
#####								EVALUATION                             #####
#####										   							   #####
#####								  PART II							   #####
#####																	   #####
################################################################################


m_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
         1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.]
learning_curve_m_ibmmac_type1 = []
learning_curve_m_ibmmac_type2 = []
learning_curve_m_sport_type1 = []
learning_curve_m_sport_type2 = []

for m_i in m_vec:
    test_labels, predicted_labels = predict(folder = "ibmmac", size = 1, m = m_i, train_type = 1)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_m_ibmmac_type1.append(acc)
for m_i in m_vec:
    test_labels, predicted_labels = predict(folder = "ibmmac", size = 1, m = m_i, train_type = 2)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_m_ibmmac_type2.append(acc)
for m_i in m_vec:
    test_labels, predicted_labels = predict(folder = "sport", size = 1, m = m_i, train_type = 1)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_m_sport_type1.append(acc)
for m_i in m_vec:
    test_labels, predicted_labels = predict(folder = "sport", size = 1, m = m_i, train_type = 2)
    acc = compute_accuracy(test_labels, predicted_labels)
    learning_curve_m_sport_type2.append(acc)

print learning_curve_m_ibmmac_type1
print learning_curve_m_ibmmac_type2
print learning_curve_m_sport_type1
print learning_curve_m_sport_type2
