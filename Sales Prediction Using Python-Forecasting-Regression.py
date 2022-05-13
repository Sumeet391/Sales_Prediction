#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Sales Prediction Using Python-Forecasting-Regression


# # Sales Prediction using Python
# Hope you now understand what sales forecasting is. Typically, a product and service-based business always need their Data Scientist to predict their future sales with every step they take to manipulate the cost of advertising their product. So let’s start the task of sales prediction with machine learning using Python. I’ll start this task by importing the necessary Python libraries and the dataset:

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[4]:


data= pd.read_csv('S://DS-22//New Projects//advertising.csv')


# In[5]:


data.head(5)


# In[6]:


data.isnull().sum()

### So this dataset does not contain any null values. Now let’s take a look at the correlation between features before 
we start training a machine learning model to predict future sales:
# In[8]:


plt.figure(figsize=(10,8))
sns.heatmap(data.corr())
plt.show()

### Now let’s prepare the data to fit into a machine learning model and then I will use a linear regression 
algorithm to train a sales prediction model using Python:
# In[9]:


x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

data = pd.DataFrame(data={"Predicted Sales": ypred.flatten()})
print(data)

