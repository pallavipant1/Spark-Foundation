#!/usr/bin/env python
# coding: utf-8

# # Prediction using Unsupervised ML

# In[84]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[46]:


data=pd.read_csv("Iris.csv")


# In[47]:


data.shape


# In[48]:


data.head()


# In[49]:


data.drop(['Id'],axis=1,inplace=True)
data


# In[50]:


data.isnull().sum() #no missing values?


# In[51]:


data.describe()


# In[52]:


sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("Iris")
g = sns.pairplot(iris)
plt.show()


# In[ ]:





# In[ ]:





# In[53]:


from sklearn.cluster import KMeans


# In[54]:


kmeans=KMeans(n_clusters=2)


# In[55]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])
data['Species'].value_counts()


# In[56]:


kmeans.fit(data)


# In[57]:


pred=kmeans.predict(data)
pred


# In[58]:


pd.Series(pred).value_counts()


# In[59]:


kmeans.inertia_


# In[60]:


kmeans.score(data)


# In[61]:


# to find the optimal number cluster and store them in SSE
SSE=[]


# In[78]:


for cluster in range(1,20):
    kmeans=KMeans(n_jobs=-1, n_clusters=cluster)
    kmeans.fit(data)
    SSE.append(kmeans.inertia_)


# In[63]:


frame=pd.DataFrame({"cluster":range(1,20),"SSE":SSE})
frame.head()


# In[64]:


plt.figure(figsize=(12,16))
plt.plot(frame["cluster"],frame["SSE"], marker=0)


# In[65]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[66]:


data_scaled=scaler.fit_transform(data)
pd.DataFrame(data_scaled).describe()


# In[67]:


SSE_scaled=[]


# In[79]:


for cluster in range (1,20):
    kmeans=KMeans(n_jobs=-1,n_clusters=cluster)
    kmeans.fit(data_scaled)
    SSE_scaled.append(kmeans.inertia_)


# In[69]:


frame_scaled=pd.DataFrame({"cluster":range(1,20),"SSE":SSE_scaled})


# In[83]:


plt.plot(frame_scaled["cluster"],frame_scaled["SSE"],marker=0)


# In[80]:


kmeans=KMeans(n_jobs=-1, n_clusters=4)
kmeans.fit(data_scaled)


# In[74]:


pred=kmeans.predict(data_scaled)


# In[75]:


frame=pd.DataFrame(data_scaled)
frame["cluster"]=pred
frame.loc[frame["cluster"]==2,:]


# In[ ]:




