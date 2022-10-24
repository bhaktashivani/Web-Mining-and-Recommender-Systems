# ## CSE 158/258, Fall 2022: Homework 1
'''
Submitted by: Shivani Bhakta 
This is CSE 258 Homework assignment. 
Credit: 
Some of the code was given as stub and some was inspired/copied from the lecture notes by McAuley, Julian and his book.  
@book{mcauley2022,
      title     = "Personalized Machine Learning",
      author    = "McAuley, Julian",
      year      = "in press",
      publisher = "Cambridge University Press"
    }
'''

import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model, metrics
import numpy as np
import random
import gzip
import math
from sklearn import svm # Library for SVM classification
import ast 


def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N



f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))




# len(dataset)
answers = {} # Put your answers to each question in this dictionary
# dataset[14]

# ### Question 1

def feature(datum):
    # your implementation
    feat = [1,datum['review_text'].count('!')]
    return feat

X = [feature(x) for x in dataset]
Y = [y['rating'] for y in dataset] # rating we are trying to predict

theta,residuals,rank,s = np.linalg.lstsq(X, Y)
theta0, theta1 = theta[0], theta[1]
mse = ((Y - np.dot(X, theta))**2).mean()

answers['Q1'] = [theta0, theta1, mse]

assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)

# ### Question 2

def feature(datum):
    feat = [1,len(datum['review_text']), datum['review_text'].count('!')]
    return feat

X = [feature(x) for x in dataset]
Y = [y['rating'] for y in dataset] # rating we are trying to predict
theta,residuals,rank,s = np.linalg.lstsq(X, Y)
theta0, theta1, theta2 = theta[0], theta[1], theta[2]
mse = ((Y - np.dot(X, theta))**2).mean()

answers['Q2'] = [theta0, theta1, theta2, mse]
assertFloatList(answers['Q2'], 4)

# ### Question 3

def feature(datum, deg):
    # feature for a specific polynomial degree
    ft = datum['review_text'].count('!')
    feat = [1]
    for i in range(1,deg+1): 
        feat.append(ft**i)
    return feat

Y = [y['rating'] for y in dataset] # rating we are trying to predict

mses = []
for i in range(1,6):
    X =[feature(x,i) for x in dataset]
    theta,residuals,rank,s = np.linalg.lstsq(X, Y)
    mses.append(((Y - np.dot(X, theta))**2).mean())

answers['Q3'] = mses

assertFloatList(answers['Q3'], 5)# List of length 5


# ### Question 4

train_data = dataset[:len(dataset)//2]
test_data = dataset[len(dataset)//2:]
train_Y = [y['rating'] for y in train_data]
test_Y = [y['rating'] for y in test_data]
mses = []
for i in range(1,6):
    # use train data
    train_X =[feature(x,i) for x in train_data]
    theta,residuals,rank,s = np.linalg.lstsq(train_X, train_Y)
    
    #mse on test data
    test_X =[feature(x,i) for x in test_data]
    mses.append(((test_Y - np.dot(test_X, theta))**2).mean())
#     print(type(theta))


answers['Q4'] = mses


assertFloatList(answers['Q4'], 5)


# ### Question 5

theta = np.median(np.array(train_Y))
mae = np.absolute((test_Y - theta)).mean() #0.907



answers['Q5'] = mae
assertFloat(answers['Q5'])


# ### Question 6

f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))

len(dataset)

def feature(datum):
    feat = [1,datum['review/text'].count('!')]
    return feat

# train_data = dataset[:len(dataset)//2]
# test_data = dataset[len(dataset)//2:]

# X_train = [feature(x) for x in train_data]
# y_train = [y['user/gender'] == 'Female' for y in train_data]

# X_test = [feature(x) for x in test_data]
# y_test = [y['user/gender'] == 'Female' for y in test_data]

X = [feature(x) for x in dataset]
y = [y['user/gender'] == 'Female' for y in dataset]


mod = linear_model.LogisticRegression()
mod.fit(X,y)

predictions = mod.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])


TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 0.5*(TPR + TNR) #balanced Error Rate 

answers['Q6'] = [TP, TN, FP, FN, BER]

assertFloatList(answers['Q6'], 5)


# ### Question 7


mod = linear_model.LogisticRegression(C=1.0, class_weight = 'balanced')
mod.fit(X,y)


predictions = mod.predict(X)


TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 0.5*(TPR + TNR) #balanced Error Rate 

answers["Q7"] = [TP, TN, FP, FN, BER]

assertFloatList(answers['Q7'], 5)


# ### Question 8

K = [1, 10, 100, 1000, 10000]
precisionList = []
for k in K: 
    mod = linear_model.LogisticRegression(C=k, class_weight = 'balanced')
    mod.fit(X,y)
    pred = mod.predict(X)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN) 
    precisionList.append(precision)

answers['Q8'] = precisionList

assertFloatList(answers['Q8'], 5) #List of five floats


f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()
