#!/usr/bin/env python
# coding: utf-8

# #  ML Assignment Groups 119 
# ## Problem Statement
# ### Predict whether the credit card using the customer is going to default or not.
# 
# <ul>
#     <li>Import the data from the  <a href="https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients">default of credit card clients</a> (2 points)</li>
# <li>Consider all columns as independent variables and assign to variable X except the last column and consider the last column as dependent variable and assign to variable y. Remove columns which don’t help the problem statement. (1 point)</li>
# <li>Compute some basic statistical details like percentile, mean, standard deviation of dataset (1 point)</li>
# <li>Do Feature Scaling on Independent variables (2 points)</li>
# <li>Split the data into train and test dataset (1 point)</li>
# <li>Use sklearn library to train on train dataset on random forest and predict on test dataset  (3 points)</li>
# <li>Compute the accuracy and confusion matrix. (2 points)</li>
# </ul>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter


# In[2]:


## Importing the dataset
df = pd.read_excel('default of credit card clients.xls',skiprows=1,usecols="B:Y")
# change the path as necessary

print(df.head())


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


scaler = StandardScaler()


# In[6]:


continous_list = ['BILL_AMT1','BILL_AMT2' ,'BILL_AMT3','BILL_AMT4','BILL_AMT5' ,'BILL_AMT6' ,'PAY_AMT1' ,'PAY_AMT2' ,'PAY_AMT3' ,'PAY_AMT4' ,'PAY_AMT5' ,'PAY_AMT6','LIMIT_BAL']


# In[7]:


continous_transform_df = pd.DataFrame(scaler.fit_transform(df[continous_list]),columns=continous_list)


# In[8]:


continous_transform_df.head()


# In[9]:


categorical_list  = np.setdiff1d(df.columns,continous_list)


# In[10]:


categorical_df = df[categorical_list]


# In[11]:


categorical_df.head()


# In[12]:


data_set = categorical_df.merge(continous_transform_df,left_index=True,right_index=True)


# In[13]:


data_set.head()


# In[14]:


X=data_set.drop('default payment next month',axis=1)
y=data_set['default payment next month']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[16]:


rf = RandomForestClassifier(random_state=42)


# In[17]:


rf.fit(X_train,y_train)


# In[18]:


y_predict=rf.predict(X_test)


# In[19]:


from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix


# In[20]:


accuracy = accuracy_score(y_pred = y_predict,y_true = y_test)


# In[21]:


accuracy


# In[22]:


conf_mat = confusion_matrix(y_pred = y_predict,y_true = y_test)


# In[23]:


conf_mat


# Checking the imbalance in the data

# In[24]:


Counter(y_test)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(data_set, annot=True, ax=ax)


# In[ ]:


plt.show()


# In[ ]:




