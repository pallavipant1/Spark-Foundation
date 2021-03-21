#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML

# In[1]:


# Predict the percentage of an student based on the no. of study hours


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


url= "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df=pd.read_csv(url)
df.head()


# In[4]:


df.describe().isnull


# In[5]:


# scatter plot for continous data


# In[6]:


df.plot(kind='scatter', x='Hours', y='Scores',figsize=(10, 6),  color='darkblue')

plt.title('Hours Vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.show()


# In[7]:


df.plot(kind="density")


# #  Implementing Linear Regression

# In[8]:


X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values  


# In[9]:


X.shape


# In[10]:


Y.shape


# In[11]:


# split the data into train and test


# In[12]:


from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                            test_size=0.25, random_state=0) 


# # import Linear regression

# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


lreg= LinearRegression()


# In[15]:


lreg.fit(X_train, Y_train)


# In[16]:


# Plotting the regression line
line=lreg.coef_*X+lreg.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# In[17]:


pred=lreg.predict(X_test)
pred


# # check performance score using r square

# In[18]:


lreg.score(X_test,Y_test)


# In[19]:


lreg.score(X_train,Y_train)


# In[20]:


print(X_test)


# In[21]:


print(Y_test)


# In[22]:


Y_pred=lreg.predict(X_test)


# In[23]:


# compare the two values the actual and predicted scores


# In[24]:


com=pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
com


# In[ ]:





# # Required Prediction for task

# In[25]:


# predict score for 9.25 hrs/day

Prediction_score = lreg.predict([[9.25]])
Prediction_score


# # Evaluation Matrix

# In[26]:



rmse_test=np.sqrt(np.mean(np.power((np.array(Y_test)-np.array(pred)),2)))
rmse_train=np.sqrt(np.mean(np.power((np.array(Y_train)-np.array(lreg.predict(X_train))),2)))
print(rmse_train)
print(rmse_test)


# In[ ]:




