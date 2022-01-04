#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.models import load_model



# In[ ]:


# Download the dataset that we choose from the kaggle web site and read the CSV file that contains all train data.


# In[3]:


train=pd.read_csv('./train.csv')


# In[ ]:


# print head of train set 


# In[6]:


train.head()


# In[8]:


train


# In[ ]:


# ckeck data types of columns and we observied all types are numberic 


# In[7]:


train.info()


# In[ ]:


#ckeck if there any missing values and result is (no missing vlaue)


# In[18]:


train.isnull().sum()


# In[26]:


train.describe().T


# In[8]:


#Check Categories for price_range
train.price_range.unique()


# In[ ]:


# we count the value of columns


# In[23]:


for col in train.columns:
    print(col + '\n-------------')
    print(train[col].value_counts())
    print('              ')
    print('-------------')


# In[ ]:


## Replace - by _ underscore
## Remove whitespace by strip
##CMake sure all are lower case


# In[11]:


train.columns = train.columns.str.replace('-', "_").str.strip().str.lower()
train.columns
train.head()


# In[13]:


train.duplicated().sum()
#No duplicates found


# In[ ]:


# Ckeck if there is any outlier


# In[30]:


sns.boxplot(x='int_memory', data=train)
sns.stripplot(x='int_memory', data=train, color="#474646")


# In[31]:


sns.boxplot(x='battery_power', data=train)
sns.stripplot(x='battery_power', data=train, color="#474646")


# ## Target

# In[44]:


sns.set_style('darkgrid')
sns.histplot(x= 'price_range', data=train)


# In[47]:


plt.figure(figsize=(190,190))
train.boxplot(by = 'price_range')
plt.show()


# In[8]:


sns.pairplot(data = train)


# In[39]:


p = sns.displot(data=train, x='price_range', stat='density', height=4, aspect=1.5)
p.fig.set_dpi(100)


# In[13]:


g = sns.FacetGrid(train, col='price_range', sharex=True, sharey=True)
g.map(sns.histplot, 'price_range',stat='density', kde=True);


# In[14]:


sns.violinplot(data=train);


# In[51]:


fig, ax = plt.subplots(1, 1, figsize=(15,8))

bar_kwargs = {'color':'tomato', 'alpha':1.0}

ax.bar(train.price_range.values, train['price_range'], label='price_range', **bar_kwargs)
ax.legend()


# In[35]:


# 1-D Chart to check Target : --> Binary Classification (0,1)
figure = plt.figure(figsize=(20,10))
sns.set(style='whitegrid', palette = 'Pastel2', font_scale=1.5)
sns.countplot(x = train.price_range , data = train);


# In[52]:


train.columns


# In[57]:


numeric_features = ['battery_power', 'blue', u'clock_speed', u'dual_sim', u'fc',
       u'four_g', u'int_memory', u'm_dep', u'mobile_wt', u'n_cores', u'pc',
       u'px_height', u'px_width', u'ram', u'sc_h', u'sc_w', u'talk_time',
       u'three_g',  'touch_screen',  'wifi',  'price_range']
#sns.color_palette("flare", as_cmap=True) # plt.pcolormesh(..., cmap="Blues")
for col in numeric_features:
    sns.histplot(x = col, data = train, hue='price_range', palette='flare', kde=True)
    plt.show()


# In[61]:


# Correlation between different variables
#
corr = train.corr()
#
# Set up the matplotlib plot configuration
#
f, ax = plt.subplots(figsize=(120, 100))
#
# Generate a mask for upper traingle
#
mask = np.triu(np.ones_like(corr, dtype=bool))
#
# Configure a custom diverging colormap
#
cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
# Draw the heatmap
#
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)


# In[ ]:





#  #### Modelling

# In[95]:


X = train.drop(columns = 'price_range')
y = train['price_range']

#Train Test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state = 42)
#plot_confusion_matrix(clf, X_test, y_test)


# In[96]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, r2_score

dt_model = DecisionTreeClassifier().fit(X_train,y_train )
dt_y_pred = dt_model.predict(X_test)

#Model Score
from sklearn.metrics import accuracy_score
print('The decision Tree model accuracy score was {}% on test dataset'.format(round(accuracy_score(dt_y_pred, y_test)*100)))
#print("Score: ", dt_model.score(X_test,y_test, average='micro')*100)
#print('The decision Tree model accuracy score was {}% on test dataset'.format(round(recall_score(y_test, dt_y_pred,average='micro')*100)))
print('recall_score ', recall_score(y_test,dt_y_pred, average='micro')*100)
print("F1 Score: ", f1_score(y_test,dt_y_pred, average='micro')*100)
print("Precision Score: ", precision_score(y_test,dt_y_pred, average='micro')*100)
print("Accuracy Score: ", accuracy_score(y_test,dt_y_pred)*100)
 


# In[100]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=dt_y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of Decision Tree', fontsize=18)
plt.show()


# In[31]:


# Bagging
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier().fit(X_train,y_train)
bag_y_pred = bag_model.predict(X_test)
print('The Bagging model accuracy score was {}% on test dataset'.format(round(accuracy_score(bag_y_pred, y_test)*100)))


# In[97]:


# Random forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(max_features=8).fit(X_train,y_train)
rf_y_pred = rf_model.predict(X_test)
print('The Random Forest model accuracy score was {}% on test dataset'.format(round(accuracy_score(rf_y_pred, y_test)*100)))
#print("Score: ", dt_model.score(X_test,y_test))
print("recall_score: ", recall_score(y_test,rf_y_pred, average='micro'))
#print("F1 Score: ", f1_score(y_test,dt_y_pred))
print("Precision Score: ", precision_score(y_test,rf_y_pred, average='micro'))
#print("Accuracy Score: ", accuracy_score(y_test,rf_y_pred)*100)


# In[99]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=rf_y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:


#read the CSV file that contains all test data.


# In[13]:


test=pd.read_csv('./test.csv')


# In[ ]:


# print test set 


# In[14]:


test


# In[ ]:




