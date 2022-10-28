#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import random as rnd

import warnings # to ignore warnings.
warnings.filterwarnings("ignore")


import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# In[15]:


train_df = pd.read_csv("C:\\Users\\tanuja\\Downloads\\train.csv")
train_df.head()


# In[38]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[17]:


a = preprocessing.LabelEncoder()
train_df['Sex'] = a.fit_transform(train_df.Sex.values)
train_df['Survived'].corr(train_df['Sex'])


# In[18]:


mat = train_df.corr()
print(mat)


# In[19]:


# 2 visualizations to show correlations.
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Sex', bins=20)


# In[20]:


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[22]:


train_df.corr().style.background_gradient(cmap="Greens")


# In[23]:


sns.heatmap(mat, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[24]:


#Implementing Naïve Bayes method using scikit-learn library and report the accuracy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[26]:


train_raw = pd.read_csv("C:\\Users\\tanuja\\Downloads\\train.csv")

test_raw = pd.read_csv("C:\\Users\\tanuja\\Downloads\\test.csv")


train_raw['train'] = 1
test_raw['train'] = 0
df = train_raw.append(test_raw, sort=False)

# Joining data to analyse and process the set as one.

features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

df = df[features + [target] + ['train']]

# Categorical values need to be converted into numeric.

df['Sex'] = df['Sex'].replace(["female", "male"], [0, 1])
df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = df.query('train == 1')
test = df.query('train == 0')


# In[27]:


# Drop missing values from the train set.
train.dropna(axis=0, inplace=True)
labels = train[target].values
train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split, cross_validate

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=0)


# In[28]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[30]:


#Implementing Naïve Bayes method using scikit-learn library.

 
glass = pd.read_csv("C:\\Users\\tanuja\\Downloads\\glass.csv")
glass.head()


# In[31]:


glass.corr().style.background_gradient(cmap="Greens")


# In[32]:


x=glass.iloc[:,:-1].values
y=glass['Type'].values


# In[33]:


#1b. Use train_test_split to create training and testing part. 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.30, random_state = 0)


# In[34]:


# Evaluating the model on testing part using score and
# 1. Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[35]:


#1. Implement linear SVM method using scikit library
#      a. Use the glass dataset available
# Support Vector Machine's 
from sklearn.svm import SVC, LinearSVC

classifier = LinearSVC()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[36]:


#Do at least two visualizations to describe or show correlations in the Glass Dataset
g = sns.FacetGrid(glass, col='Type')
g.map(plt.hist,'RI',bins=20)


# In[37]:


grid = sns.FacetGrid(glass, row='Type',col='Ba',height=2.2,aspect=1.6)
grid.map(sns.barplot,'Al','Ca',alpha=.5,ci=None)
grid.add_legend()


# In[ ]:


#Which algorithm you got better accuracy? Can you justify why?
#Gaussian Naive Bayes algorithm gives better accuracy than other algorithms. This is used when features are not discreet.
