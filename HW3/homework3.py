#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import numpy as np 


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[4]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[5]:


answers = {}


# In[6]:


# Some data structures that will be useful


# In[7]:


allRatings = []
for l in readCSV("assignment1/train_Interactions.csv.gz"):
    allRatings.append(l)


# In[8]:


len(allRatings)


# In[9]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
num_users = len(ratingsPerUser)
num_items = len(ratingsPerItem)


# In[10]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[11]:


# get labels 
ratingsLabels = [d[-1] for d in ratingsTrain]
globalAverage = sum([d[-1] for d in ratingsTrain]) / len(ratingsTrain)
users = list(ratingsPerUser.keys())
items = list(ratingsPerItem.keys())


# In[12]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[13]:


# define mean and bias terms as global variables 
alpha = globalAverage
betaUser = defaultdict(float)
betaItem = defaultdict(float)
reg_lambda = 1 

# update mean and bias iteratively as described in lecture notes update equations.  
def update_params(param):
    global alpha
    global betaUser
    global betaItem
    
    alpha = param[0]
    betaUser = dict(zip( users, param[ 1 : (num_users+1) ]) )
    betaItem = dict(zip( items, param[ (1+num_users) : ]) )


# In[14]:


def predict(user, item):
    pred = alpha + betaUser[user] + betaItem[item]
    return pred


# In[15]:


def getMSE(param, labels):
    
    update_params(param) # first update the parameters.
    ypred = [predict(d[0], d[1]) for d in ratingsTrain]
    
    mse = MSE(ypred, labels) #error 
    print("MSE = " + str(mse))
    
    # error + regularizer 
    for u in betaUser:
        mse += reg_lambda * betaUser[u]**2
        
    for i in betaItem:
        mse += reg_lambda * betaItem[i]**2
        
    return mse 


# In[16]:


def gradiant(param, labels):
    
    update_params(param) # first update the parameters. 
    

    Ntrain = len(ratingsTrain)
    
    
    #temp var 
    a = 0
    betaU = defaultdict(float)
    betaI = defaultdict(float)
    
    for d in ratingsTrain:
        
        user,item = d[0], d[1]
        ypred = predict(user, item)
        diff = ypred - d[-1]
        
        a += 2/Ntrain * diff
        betaU[user] += 2/Ntrain * diff
        betaI[item] += 2/Ntrain * diff
    
    for u in betaUser:
        betaU[u] += 2*reg_lambda*betaUser[u]
    for i in betaItem:
        betaI[i] += 2*reg_lambda*betaItem[i]
    
    param = numpy.array([a] + [betaU[u] for u in users] + [betaI[i] for i in items])
    
    return param


# In[17]:


### Question 9


# In[18]:


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
scipy.optimize.fmin_l_bfgs_b(getMSE, [alpha] + [0.0]*(num_users+num_items),
                             gradiant, args = ([ratingsLabels]))


# In[19]:


ypred = []
valid_labels = []
for d in ratingsValid:
    user, item = d[0], d[1]
    valid_labels.append(d[2])
    if user in betaUser and item in betaItem:
        ypred.append(predict(user, item))
    else: ypred.append(0)

print("MSE = %f" % MSE(ypred, valid_labels))
validMSE = MSE(ypred, valid_labels)


# In[20]:


answers['Q9'] = validMSE


# In[21]:


assertFloat(answers['Q9'])


# In[22]:


### Question 10


# In[23]:


maxUser = max(betaUser, key=betaUser.get)
maxBeta = float(betaUser[maxUser])
minUser = min(betaUser, key=betaUser.get)
minBeta = float(betaUser[minUser])


# In[24]:


answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]


# In[25]:


assert [type(x) for x in answers['Q10']] == [str, str, float, float]


# In[26]:


### Question 11


# In[27]:


def training(lamb):
    validMSEs = []
    for l in lamb: 
        global reg_lambda
        reg_lambda = l 
        scipy.optimize.fmin_l_bfgs_b(getMSE, [alpha] + [0.0]*(num_users+num_items),
                             gradiant, args = ([ratingsLabels]))
        
        ypred = []
        valid_labels = []
        for d in ratingsValid:
            user, item = d[0], d[1]
            valid_labels.append(d[2])
            if user in betaUser and item in betaItem:
                ypred.append(predict(user, item))
            else: ypred.append(0)

        print("For lamba: {}, MSE = {}" .format(reg_lambda,MSE(ypred, valid_labels)))
        tmse = MSE(ypred, valid_labels)
        validMSEs.append(tmse)
    return validMSEs, lamb
validMSEs, lamb = training([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]) 


# In[28]:


idx = validMSEs.index(min(validMSEs))
validMSE = min(validMSEs)
lamb = lamb[idx]
print(lamb, validMSE)


# In[29]:


answers['Q11'] = (lamb, validMSE)


# In[30]:


assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][0])


# In[31]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("assignment1/pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()


# In[32]:


##################################################
# Read prediction                                #
##################################################


# In[33]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("assignment1/train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[34]:


### Question 1


# In[35]:


# get all the books user has read 
itemsPerUser = {}
for d in allRatings:
    u = d[0]
    i = d[1]
    if u in itemsPerUser:
        itemsPerUser[u].add(i)
    else:
        itemsPerUser[u] = {i}

# add negative samples for each user 
negValidRatings = [[d[0],d[1],d[2],1] for d in ratingsValid]

temp = np.array(negValidRatings)
all_books = set(temp[:,1]) #set helps to get unique book_ids

for d in ratingsValid:
    unread_books = all_books - set(itemsPerUser)
    bookid = random.sample(unread_books, 1)[0]
    
    # add this random book that this user has not read/rated yet with label 0 and rating 0
    negValidRatings.append([d[0], bookid, 0, 0])


# In[36]:


acc1 = 0
for d in negValidRatings:
    if d[1] not in return1:
        acc1 += d[-1]==0
    else:
        acc1 += d[-1]==1
acc1 /= len(negValidRatings)
acc1


# In[37]:


answers['Q1'] = acc1


# In[38]:


assertFloat(answers['Q1'])


# In[39]:


### Question 2


# In[40]:


# same code as earlier but we want to test different threasholds that might work better than just the average one. 
# Copied from baseline code

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("assignment1/train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

thresholds = [i/10 for i in range(10,30)]
accPerThreshold = {}


for thred in thresholds: 
    
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/thred: break
    
    acc1 = 0
    for d in negValidRatings:
        if d[1] not in return1:
            acc1 += d[-1]==0
        else:
            acc1 += d[-1]==1
    acc1 /= len(negValidRatings)
    accPerThreshold[thred]=acc1

import matplotlib.pyplot as plt
plt.plot(accPerThreshold.keys(), accPerThreshold.values())
plt.show()


# In[41]:


acc2 = max(accPerThreshold.values())
threshold = max(accPerThreshold, key=accPerThreshold.get)
print([threshold, acc2])


# In[42]:


answers['Q2'] = [threshold, acc2]


# In[43]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[44]:


### Question 3/4


# In[45]:


# copied from last HW 
def Jaccard(s1, s2):
    num = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    
    if denom == 0:
        return 0
    return num/denom  


# In[46]:


# Get all the pairs 
booksPerUser = defaultdict(set)
usersPerbook = defaultdict(set)

for d in ratingsTrain:
    u = d[0]
    i = d[1]
    r = d[-1]
    if u in booksPerUser:
        booksPerUser[u].add(i)
    else:
        booksPerUser[u] = {i}
        
    if i in usersPerbook:
        usersPerbook[i].add(u)
    else:
        usersPerbook[i] = {u}


# In[47]:


def predictRead(threshold):
    acc0 = 0
    for d in negValidRatings: # pairs in validation set 
        u = d[0]
        b = d[1]
        books = booksPerUser[d[0]] # consider all training items b′ that user u has read 
        similarities = []
        if books: #if user has read any book previously 
            for b1 in books:
                if(b==b1):
                    continue
                if b in usersPerbook:
                    sim = Jaccard(usersPerbook[b], usersPerbook[b1])
                    similarities.append(sim)
                else:
                    similarities.append(0) #no similarity 

            isRead = d[-1]==1
            isNotRead = d[-1]==0
            
            if (threshold < max(similarities)):
                acc0 += isRead
            else:
                acc0 += isNotRead
    acc0 /=len(negValidRatings)
    
    return acc0


# In[48]:


thresholds = []
for i in range(1,15): 
    thresholds.append(1/(2**i))
thresholds


# In[49]:


accPerThreshold = {}
for thred in thresholds:
    accPerThreshold[thred] = predictRead(thred)
    print("For threshold: {}, accuracy = {}" .format(thred, accPerThreshold[thred]))

acc3 = max(accPerThreshold.values())
threshold = max(accPerThreshold, key=accPerThreshold.get)
print([threshold, acc3])


# In[50]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("assignment1/train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/answers['Q2'][0]: break


# In[51]:


answers['Q2']


# In[52]:


def predictRead(threshold):
    acc0 = 0
    for d in negValidRatings: # pairs in validation set 
        u = d[0]
        b = d[1]
        books = booksPerUser[d[0]] # consider all training items b′ that user u has read 
        similarities = []
        if books: #if user has read any book previously 
            for b1 in books:
                if(b==b1):
                    continue
                if b in usersPerbook:
                    sim = Jaccard(usersPerbook[b], usersPerbook[b1])
                    similarities.append(sim)
                else:
                    similarities.append(0) #no similarity 

            isRead = d[-1]==1
            isNotRead = d[-1]==0
            
            if (threshold < max(similarities)) and b in return1 :
                acc0 += isRead
            else:
                acc0 += isNotRead
    acc0 /=len(negValidRatings)
    
    return acc0


acc4 = predictRead(threshold)


# In[53]:


answers['Q3'] = acc3
answers['Q4'] = acc4


# In[54]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[55]:


predictions = open("predictions_Read.csv", 'w')
for l in open("assignment1/pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[56]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[57]:


assert type(answers['Q5']) == str


# In[58]:


answers


# In[59]:


# Run on test set


# In[60]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("assignment1/pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[61]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




