## CSE 158/258, Fall 2022: Homework 1
'''
Submitted by: Shivani Bhakta 
This is CSE 258 Midterm. 
Credit: 
Some of the code was given as stub and some was inspired/copied from the lecture notes by McAuley, Julian and his book.  
@book{mcauley2022,
      title     = "Personalized Machine Learning",
      author    = "McAuley, Julian",
      year      = "in press",
      publisher = "Cambridge University Press"
    }, 
    
  and some code was also taken from my own homework assignments submitted in this course earlier. 
'''

# In[1]:


import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model


# In[2]:


# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


answers = {}


# In[5]:


f = open("spoilers.json.gz", 'r')


# In[6]:


dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)


# In[7]:


f.close()


# In[8]:


len(dataset)


# In[9]:


dataset[2]


# In[10]:


# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
# ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataset:
    u,i = d['user_id'],d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
#     ratingDict[(u,i)] = d['rating'] 
    
    
# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])
    
# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])


# In[11]:


# E.g. reviews for this user are sorted from earliest to most recent
[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# ## 1a

# In[12]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[13]:


# remove any user who has only 1 or less reviews
user_1 = [] 
for key, value in reviewsPerUser.items(): 
    if len(value) <= 1: 
        user_1.append(key)
[reviewsPerUser.pop(u) for u in user_1]
# len(reviewsPerUser)

# remove any items who has only 1 or less reviews
items_1 = [] 
for key, value in reviewsPerItem.items(): 
    if len(value) <= 1: 
        items_1.append(key)
[reviewsPerItem.pop(i) for i in items_1]
len(reviewsPerItem)


# In[14]:


# Get the averages 
userAverages = {}
itemAverages = {}

y = []
for u in reviewsPerUser:
    rs = [i['rating'] for i in reviewsPerUser[u]]
    y.append(rs[-1])
    userAverages[u] = sum(rs[:-1]) / (len(rs)-1)  


# In[15]:


ypred = [userAverages[u] for u in userAverages]
# len(ypred)


# In[16]:


answers['Q1a'] = MSE(y,ypred)


# In[17]:


assertFloat(answers['Q1a'])


# ## 1b

# In[18]:


y = []
for i in reviewsPerItem:
    rs = [j['rating'] for j in reviewsPerItem[i]]
    y.append(rs[-1])
    itemAverages[i] = sum(rs[:-1]) / (len(rs)-1)
ypred = [itemAverages[u] for u in itemAverages]
# len(ypred)


# In[19]:


answers['Q1b'] = MSE(y,ypred)


# In[20]:


assertFloat(answers['Q1b'])


# ## 2

# In[21]:


def computeAvg(reviews,N): 
    # Get the averages 
    Averages = {}
    
    y = []
    for u in reviews:
        rs = [i['rating'] for i in reviews[u]]
        y.append(rs[-1])
        if len(rs) < N+1:
    
            Averages[u] = sum(rs[:-1]) / (len(rs)-1)    
        else:
            lastN = rs[-(N+1):][:-1]
            Averages[u] = sum(lastN) / (len(lastN))  
            
    return y,Averages


# In[22]:


answers['Q2'] = []

for N in [1,2,3]:
    y, userAverages = computeAvg(reviewsPerUser, N)
    ypred = [userAverages[u] for u in userAverages]
    answers['Q2'].append(MSE(y,ypred))


# In[23]:


assertFloatList(answers['Q2'], 3)


# ## 3a

# In[24]:


# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])


# In[25]:


def feature3(N, u): # For a user u and a window size of N
    rs = [i['rating'] for i in reviewsPerUser[u]]
#     print(rs)
#     lastN = rs[-(N+1):][:-1]
    lastN = rs[-(N+1):-1]
    lastN.reverse()
    feat = [1] + lastN
    
    return feat


# In[26]:


answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]


# In[27]:


assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4


# ## 3b

# In[28]:


import numpy as np
answers['Q3b'] = []

for N in [1,2,3]:
    X = []
    Y = []
    for u in reviewsPerUser: 
        rs = [ i['rating'] for i in reviewsPerUser[u] ]
        x = feature3(N, u)
        if len(x) < N+1: 
            continue 
        X.append(x)
        Y.append(rs[-1])
        
    theta,residuals,rank,s = np.linalg.lstsq(X, Y)
    mse = ((Y - np.dot(X, theta))**2).mean()
    answers['Q3b'].append(mse)
# answers['Q3b'].append(0)
# answers['Q3b'].append(0)
answers['Q3b']
# print(len(X), len(Y))


# In[29]:


assertFloatList(answers['Q3b'], 3)


# ## 4a

# In[30]:


globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)


# In[31]:


def featureMeanValue(N, u): # For a user u and a window size of N
    
    rs = [i['rating'] for i in reviewsPerUser[u]]
    if len(rs) < N+1:
        Averages = sum(rs[:-1]) / (len(rs)-1) 
        x = [Averages]* (N+1 - len(rs))
        X = feature3(N, u) + x
    else:
        X = feature3(N, u) 

    return X

# featureMeanValue(10, dataset[0]['user_id'])


# In[32]:


def featureMissingValue(N, u):
    
    rs = [i['rating'] for i in reviewsPerUser[u]]

    f4 = [1]
    for f in range(1,len(feature3(N, u))):
        f4.append(0) 
        f4.append(feature3(N, u)[f]) 

    if len(rs) < N+1:
        x = [1,0]* (N+1 - len(rs))
        X = f4 + x
    else:
        X= f4  
            
    return X

# featureMissingValue(10, dataset[0]['user_id'])


# In[33]:


answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]


# In[34]:


# answers['Q4a']


# In[35]:


assert len(answers['Q4a']) == 2
assert len(answers['Q4a'][0]) == 11
assert len(answers['Q4a'][1]) == 21


# ## 4b

# In[36]:


answers['Q4b'] = []
X = []
Y = []
N = 10
for u in reviewsPerUser: 
    rs = [ i['rating'] for i in reviewsPerUser[u] ]
    X.append(featureMeanValue(N, u))
    Y.append(rs[-1])

theta,residuals,rank,s = np.linalg.lstsq(X, Y)
mse = ((Y - np.dot(X, theta))**2).mean()
answers['Q4b'].append(mse)


# In[37]:


X = []
Y = []
N = 10
for u in reviewsPerUser: 
    rs = [ i['rating'] for i in reviewsPerUser[u] ]
    X.append(featureMissingValue(N, u))
    Y.append(rs[-1])

theta,residuals,rank,s = np.linalg.lstsq(X, Y)
mse = ((Y - np.dot(X, theta))**2).mean()
answers['Q4b'].append(mse)


# In[38]:


# answers['Q4b'] = []

# for featFunc in [featureMeanValue, featureMissingValue]:
#     # etc.
    
#     answers['Q4b'].append(mse)


# In[39]:


assertFloatList(answers["Q4b"], 2)


# ## 5

# In[40]:


def feature5(sentence):
    f3 = [1 for c in sentence if c.isupper()]
    feat = [1,len(sentence), sentence.count('!'), sum(f3)]
    return feat


# In[41]:


def BER_(predictions, y):
    # Balance Error Rate function from HW2 
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    BER = 1 - 0.5*(TPR + TNR) #balanced Error Rate 
    return TP, TN, FP, FN, BER


# In[42]:


y = []
X = []

for d in dataset:
    for spoiler,sentence in d['review_sentences']:
        X.append(feature5(sentence))
        y.append(spoiler)


# In[43]:


mod = linear_model.LogisticRegression(C=1.0, class_weight="balanced")
mod.fit(X, y)

predictions = mod.predict(X)


# In[44]:


TP, TN, FP, FN, BER = BER_(predictions, y)


# In[45]:


answers['Q5a'] = X[0]


# In[46]:


answers['Q5b'] = [TP, TN, FP, FN, BER]


# In[47]:


assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)


# ## 6

# In[48]:


def feature6(review):
    X = [1]
    N = 5
    for N in range(len(review)):
        if N < 5: 
            spoiler = review[N][0]
            X.append(spoiler)
        elif N == 5: 
            X = X + feature5(review[N][1])[1:]
    return X


# In[49]:


y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(sentences))
    y.append(sentences[5][0])


# In[50]:


X[0]


# In[51]:


answers['Q6a'] = X[0]


# In[52]:


mod = linear_model.LogisticRegression(C=1.0, class_weight="balanced")
mod.fit(X, y)

predictions = mod.predict(X)

TP, TN, FP, FN, BER = BER_(predictions, y)


# In[53]:


answers['Q6b'] = BER


# In[54]:


assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])


# ## 7

# In[55]:


# 50/25/25% train/valid/test split
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[56]:


## Got most of this code from Chapter 3 workbook and edited it as needed. 

def pipeline(reg):
    mod = linear_model.LogisticRegression(C=reg, class_weight='balanced')
        
    mod.fit(Xtrain,ytrain)
    ypredValid = mod.predict(Xvalid)
    ypredTest = mod.predict(Xtest)
    
    # validation
    
    TP, TN, FP, FN, berVal = BER_(ypredValid,yvalid)
    print("C = " + str(reg) + "; validation BER = " + str(berVal))
    
    # test

    TP, TN, FP, FN, berTest = BER_(ypredTest,ytest)
#     print("C = " + str(reg) + "; test BER = " + str(berTest))

    return mod, berVal, berTest


# In[57]:


modList = []
berList = []
berTestList = []

for c in [0.01, 0.1, 1, 10, 100]:
    # etc.
    mod, berVal, berTest = pipeline(c)
    modList.append(mod)
    berList.append(berVal) 
    berTestList.append(berTest)
bers = berList


# In[58]:


# Since C=0.1 has the smallest validation error we choose that as a general good performer 
bestC = 0.1
ber = berTestList[1]


# In[59]:


answers['Q7'] = bers + [bestC] + [ber]


# In[60]:


assertFloatList(answers['Q7'], 7)


# ## 8

# In[61]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[62]:


# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]
len(dataTrain),len(dataTest)


# In[63]:


# A few utilities

userAverages = defaultdict(list)
itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    
    userAverages[d['user_id']].append(d['rating'])
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

# for i,u in zip(itemAverages, userAverages):
#     itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])
#     userAverages[u] = sum(userAverages[u]) / len(userAverages[u])
for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])
    
for u in userAverages:
    userAverages[u] = sum(userAverages[u]) / len(userAverages[u])

ratingMean = sum(ratingMean) / len(ratingMean)


# In[64]:


# reviewsPerItem = defaultdict(list)
itemsPerUser = defaultdict(set) 

reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    u,i = d['user_id'], d['book_id'] 
    
    usersPerItem[i].add(u)
    itemsPerUser[u].add(i)
    
    reviewsPerUser[u].append(d)
#     reviewsPerItem[i].append(d)
    ratingDict[(u,i)] = d['rating'] 


# In[65]:


def predictRating(user,item):
    
    ratings = []
    similarities = []
    
    for i2 in usersPerItem[item]:
        
        if i2 == user: continue
        
        ratings.append(ratingDict[(i2,item)] - userAverages[i2])
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[i2]))
        
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean


# In[66]:


predictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]


# In[67]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[68]:


answers["Q8"] = MSE(predictions, labels)
answers["Q8"]


# In[69]:


assertFloat(answers["Q8"])


# ## 9a

# In[70]:


usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id'] 
    usersPerItem[i].add(u)


# In[71]:


# MSE for instances where i never appeared in the training set
itemsTrain = set(usersPerItem.keys())
# print(itemsTrain)
itemsTest_unique = []

uniqueItemsRating = []

for d in dataTest:
#     print(d['book_id'])
    if d['book_id'] not in itemsTrain: 
        itemsTest_unique.append(d)
#         print(itemsTest_unique)
        uniqueItemsRating.append(d['rating'])
predictions = [predictRating(d['user_id'], d['book_id']) for d in itemsTest_unique]
labels = [r for r in uniqueItemsRating]
mse0 = MSE(predictions, labels)  
mse0


# In[72]:


# MSE for instances where i appeared in the training set for atleast once but no more than 5 times. 
itemsTrain1to5 = set()
itemsTrain6plus = set()

for i in usersPerItem: 
    num_reviewed = len(usersPerItem[i])
    if num_reviewed > 5: 
        itemsTrain6plus.add(i)  
    else: 
        itemsTrain1to5.add(i)
       
    
itemsTest1to5 = []
itemsTestRating1to5 = []


for d in dataTest:
    if d['book_id'] in itemsTrain1to5:
        itemsTest1to5.append(d)
        itemsTestRating1to5.append(d['rating'])       
        
predictions = [predictRating(d['user_id'], d['book_id']) for d in itemsTest1to5]
labels = [r for r in itemsTestRating1to5 ]
mse1to5 = MSE(predictions, labels)


# In[73]:


items_in_test_more_5_train = []
r_items_in_test_more_5_train = []

for d in dataTest:
    if d['book_id'] in itemsTrain6plus:
        items_in_test_more_5_train.append(d)
        r_items_in_test_more_5_train.append(d['rating'])
        
predictions = [predictRating(d['user_id'], d['book_id']) for d in items_in_test_more_5_train]
labels = [r for r in r_items_in_test_more_5_train]
mse5 = MSE(predictions, labels)


# In[74]:


mse1to5


# In[75]:


mse5


# In[76]:


answers["Q9"] = [mse0, mse1to5, mse5]


# In[77]:


assertFloatList(answers["Q9"], 3)


# ## 10

# In[78]:


usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id'] 
    usersPerItem[i].add(u)  


# In[79]:


def NewpredictRating(user,item):
    
    ratings = []
    similarities = []
    
    for d in reviewsPerUser[item]:
        i2 = d['book_id']
        if i2 == user: continue
#         ratingDict[(i2,item)]
        ratings.append(ratingDict[(i2,item)] - itemAverages[i2])
        
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
        
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    
    else:
        # User hasn't rated any similar items
            
        if item in itemAverages:
            return itemAverages[item]
        elif user in userAverages:
            return userAverages[user]
        else:
            return ratingMean


# In[80]:


# MSE for instances where i never appeared in the training set
itemsTrain = set(usersPerItem.keys())
# print(itemsTrain)
itemsTest_unique = []

uniqueItemsRating = []

for d in dataTest:
#     print(d['book_id'])
    if d['book_id'] not in itemsTrain: 
        itemsTest_unique.append(d)
        uniqueItemsRating.append(d['rating'])
predictions = [NewpredictRating(d['user_id'], d['book_id']) for d in itemsTest_unique]
labels = [r for r in uniqueItemsRating]
itsMSE = MSE(predictions, labels)  
itsMSE


# In[81]:


Solution = "If we don't have the item in the training set, we can take the average rating for the given user for other items and return that as a prediction. Therefore, instead of prediction mean of all item, we are getting a geenral idea of how a specific user tends to rate their products. This slightly performs better as there is some correlation there, however it is not much of a difference as expected. It is still in the range from part a to part c MSE from previous part"


# In[82]:


answers["Q10"] = (Solution, itsMSE)


# In[83]:


assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])


# In[84]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




