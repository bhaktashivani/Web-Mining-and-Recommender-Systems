## CSE 158/258, Fall 2022: Homework 1
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


import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
# from pathlib import Path
# import sys
# sys.path.append("../")
# sys.path.append("../data")



def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


f = open("data/5year.arff", 'r')


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]



answers = {} # Your answers


def accuracy(predictions, y):
    acc = (predictions == y)
    return sum(acc)/len(acc)
    

def BER(predictions, y):
    
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    BER = 1 - 0.5*(TPR + TNR) #balanced Error Rate 
    return BER



### Question 1




mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)

acc1 = accuracy(pred, y)
ber1 = BER(pred, y)


answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate

assertFloatList(answers['Q1'], 2)


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)

acc2 = accuracy(pred, y)
ber2 = BER(pred, y)

answers['Q2'] = [acc2, ber2]

assertFloatList(answers['Q2'], 2)


### Question 3

random.seed(3)
random.shuffle(dataset)

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

len(Xtrain), len(Xvalid), len(Xtest)

mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)

ypredtrain = mod.predict(Xtrain)
ypredvalid = mod.predict(Xvalid)
ypredTest = mod.predict(Xtest)

berTrain = BER(ypredtrain, ytrain)
berValid = BER(ypredvalid, yvalid)
berTest = BER(ypredTest, ytest)

answers['Q3'] = [berTrain, berValid, berTest]

assertFloatList(answers['Q3'], 3)

### Question 4

## Got most of this code from Chapter 3 workbook and edited it as needed. 

def pipeline(reg):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')
        
    mod.fit(Xtrain,ytrain)
    ypredValid = mod.predict(Xvalid)
    ypredTest = mod.predict(Xtest)
    
    # validation
    
    berVal = BER(ypredValid,yvalid)
    print("C = " + str(reg) + "; validation BER = " + str(berVal))
    
    # test

    berTest = BER(ypredTest,ytest)
#     print("C = " + str(reg) + "; test BER = " + str(berTest))

    return mod, berVal, berTest

modList = []
berList = []
berTestList = []
for c in [1e-04, 1e-03, 1e-02, 1e-01, 1, 10, 100, 1000, 10000]:
    mod, berVal, berTest = pipeline(c)
    modList.append(mod)
    berList.append(berVal) 
    berTestList.append(berTest)

answers['Q4'] = berList

assertFloatList(answers['Q4'], 9)

### Question 5


# Since C=100 has the smallest validation error we choose that as a general good performer 
bestC = 100
ber5 = berTestList[6]

answers['Q5'] = [bestC, ber5]

assertFloatList(answers['Q5'], 2)

### Question 6

f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))


dataTrain = dataset[:9000]
dataTest = dataset[9000:]

len(dataTrain), len(dataTest)

# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user, item, reviews = d['user_id'], d['book_id'], d['review_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    
    
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)

def Jaccard(s1, s2):
    num = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    
    if denom == 0:
        return 0
    return num/denom

def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    
    candidateItems = set()
    for u in users:
        candidateItems = candidateItems.union(itemsPerUser[u])
        
    for ii in candidateItems: #usersPerItem:
        if ii == i: continue
        sim = Jaccard(users, usersPerItem[ii])
        similarities.append((sim,ii))
    similarities.sort(reverse=True)
    return similarities[:N]

answers['Q6'] = mostSimilar('2767052', 10)


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)

### Question 7

userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)

def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        
        if i2 == item: continue
        
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
        
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)\
    
    else:
        # User hasn't rated any similar items
        ratingMean = sum([d['rating'] for d in dataTrain]) / len(dataTrain)
        return ratingMean

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]

labels = [d['rating'] for d in dataTest]
mse7 = MSE(simPredictions, labels)


answers['Q7'] = mse7


assertFloat(answers['Q7'])


### Question 8


def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerItem[item]:
        u2 = d['user_id']
        
        if u2 == user: continue
        
        ratings.append(d['rating'])
        similarities.append(Jaccard(itemsPerUser[item],itemsPerUser[u2]))
        
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    
    else:
        # User hasn't rated any similar items
        ratingMean = sum([d['rating'] for d in dataTrain]) / len(dataTrain)
        return ratingMean


simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]


labels = [d['rating'] for d in dataTest]
mse8 = MSE(simPredictions, labels)

answers['Q8'] = mse8


assertFloat(answers['Q8'])


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


