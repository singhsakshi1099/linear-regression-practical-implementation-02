#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


from sklearn.datasets import load_boston


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


df = load_boston()


# In[9]:


df


# In[12]:


dataset=pd.DataFrame(df.data)


# In[13]:


dataset


# In[16]:


dataset.columns=df.feature_names


# In[17]:


dataset.head()


# In[18]:


x=dataset
y=df.target


# In[19]:


y


# In[28]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
     x, y, test_size=0.30, random_state=42) 


# In[29]:


x_train


# In[30]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[31]:


x_train=scaler.fit_transform(x_train)


# In[32]:


x_test=scaler.transform(x_test)


# In[33]:


from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score


# In[34]:


regression=LinearRegression()
regression.fit(x_train,y_train)


# In[39]:


mse=cross_val_score(regression,x_train,y_train,scoring='neg_mean_squared_error',cv=10)


# In[40]:


np.mean(mse)


# In[41]:


reg_pred=regression.predict(x_test)


# In[42]:


reg_pred


# In[43]:


import seaborn as sns
sns.displot(reg_pred-y_test,kind='kde')


# In[44]:


from sklearn.metrics import r2_score


# In[46]:


score=r2_score(reg_pred,y_test)

score
# In[ ]:




