#!/usr/bin/env python
# coding: utf-8

# #Importing the Libraries

# In[45]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns


# #Loading the data set

# In[12]:


df=pd.read_csv('MagicBricks.csv') 
df.head(10)


# In[13]:


df.columns


# In[14]:


df.info()


# In[15]:


df.describe()


# In[16]:


df.head()


# In[17]:


df.tail()


# In[18]:


df.shape


# In[19]:


df.isna().sum()


# In[20]:


df.isna().mean()*100


# In[21]:


df.dtypes


# In[22]:


df["Price"].value_counts()


# In[23]:


df["Price"].nunique()


# In[24]:


df["Status"].nunique()


# In[25]:


df["Transaction"].nunique()


# In[26]:


df["Locality"].value_counts()


# # here we can see that the count of int64 is more than the float64 and object

# In[46]:


sns.countplot(df.dtypes.map(str))
plt.show()


# In[30]:


df.groupby("Locality").agg([np.median])


# In[39]:


df.columns


# In[40]:


df["BHK"].unique()


# In[41]:


df["BHK"].value_counts()


# In[42]:


import warnings
warnings.filterwarnings("ignore")


# In[47]:


for index, i in enumerate(df.columns):
        if(df[i].dtype == np.float64 or df[i].dtype == np.int64):
              plt.figure(index)
              sns.boxplot(df[i])
plt.show()


# In[48]:


for index, i in enumerate(df.columns):
        if(df[i].dtype == np.float64 or df[i].dtype == np.int64):
              plt.figure(index)
              sns.distplot(df[i])
plt.show()


# In[49]:


for index, i in enumerate(df.columns):
        if(df[i].dtype == np.float64 or df[i].dtype == np.int64):
              plt.figure(index)
              sns.kdeplot(df[i])
plt.show()


# In[50]:


for index, i in enumerate(df.columns):
        if(df[i].dtype == np.float64 or df[i].dtype == np.int64):
              plt.figure(index)
              sns.violinplot(df[i])
plt.show()


# In[51]:


sns.pairplot(df,hue="Price")


# In[52]:


df.corr()


# In[53]:


plt.figure(figsize=(15,8))
sns.heatmap(data=df.corr(),annot=True)
plt.show()


# In[54]:


count=1
plt.subplots(figsize=(15,10))
for i in df.columns:
    if df[i].dtypes!="object":
        plt.subplot(3,2,count)
        sns.distplot(df[i])
        count+=1
plt.show()


# In[75]:


X=df.drop('Price',axis=1)
Y=df['Price']


# In[76]:


X.shape


# In[77]:


Y.shape


# #splitting test data and train train

# In[78]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)


# In[79]:


X_train


# In[81]:


y_train


# In[82]:


X_test


# In[83]:


y_test


# In[116]:


df.columns


# In[132]:


#Correlation matrix
# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[133]:


# Let's see the correlation matrix 
plt.figure(figsize = (16,10))     # Size of the figure
sns.heatmap(df.corr(),annot = True)


# In[ ]:





# In[145]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf.fit(X_train, y_train)


# In[144]:


y_pred_rf = rf.predict(x_test)
y_pred_rf


# In[ ]:


#display adjusted R-squared
a_r_s=(1 - (1-rf.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1))*100
print('adjusted R-squared:', a_r_s)

