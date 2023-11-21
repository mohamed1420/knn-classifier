#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics


# In[96]:


bank=pd.read_csv('Downloads\BankNote_Authentication.csv')


# In[97]:


bank.head()


# # Normalization

# In[98]:


df = bank.drop(["class"],axis=1)
df_norm = ((df-df.mean())/df.std())
df_norm = pd.concat((df_norm, bank['class']), 1)
df_norm.head()


# In[99]:


x =df_norm.drop(["class"],axis=1) #df_norm.iloc[:,:4].values
y = df_norm['class'] # df_norm.iloc[:,-1].values


# In[100]:


x=x.to_numpy()
y=y.to_numpy()


# # splitting data

# In[101]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)


# In[102]:


print(x.shape, x_train.shape, x_test.shape)


# In[103]:


trainning_data = np.insert(x_train, 4, y_train, axis=1)
trainning_data.shape


# In[104]:


def accuracy(y_true, y_predicted):
    accuracy = np.sum(y_true == y_predicted)/len(y_true)
    return accuracy


# # Knn class

# In[115]:


class KNN_Classifier():

  
  def get_distance_metric(self,training_data_point, test_data_point):
        
    dist = 0
    for i in range(len(training_data_point) - 1):
        dist = dist + (training_data_point[i] - test_data_point[i])**2

    euclidean_dist = np.sqrt(dist)
    
    return euclidean_dist


  def nearest_neighbors(self,X_train, test_data, k):

    distance_list = []

    for training_data in X_train:
        distance = self.get_distance_metric(training_data, test_data)
        distance_list.append((training_data, distance))

    distance_list.sort(key=lambda x: x[1])
    
    neighbors_list = []

    for j in range(k):
        neighbors_list.append(distance_list[j][0])

    return neighbors_list


  def predict(self, X_train, test_data, k):
    neighbors = self.nearest_neighbors(X_train, test_data, k)
    
    for data in neighbors:
        label = []
        label.append(data[-1])

    predicted_class = statistics.mode(label)

    return predicted_class


# In[116]:


KNN=KNN_Classifier()


# In[117]:


def nums(first_number, last_number, step=1):
    return range(first_number, last_number+1, step)


# In[118]:



for k in nums(1,9):
    tr =0
    for i in range(len(x_test)):
        p = KNN.predict(trainning_data,x_test[i],k)
        if(p==y_test[i]):
            tr=tr+1   
    print("K value :" ,k )
    print("Number of correctly classified instances : ",tr,"Total number of instances : ", len(y_test))
    print("Accuracy :" ,(tr/len(y_test))*100,"%")


# In[ ]:


zero = 0
    one = 0
    for data in neighbors:
      
        if data[-1]==0:
            zero = zero + 1
        elif data[-1]==1:
            one = one + 1
    if one == zero:
        print("equal")
        self.predict(X_train,test_data,k+1)
    #print(label)

