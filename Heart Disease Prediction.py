#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#There are 14 features(Columns) including the target. The data set includes features like:

#slope_of_peak_exercise_st_segment (type: int): the slope of the peak exercise ST segment, an electrocardiography read out indicating quality of blood flow to the heart

#thal (type: categorical): results of thallium stress test measuring blood flow to the heart, with possible values normal, fixed_defect, reversible_defect

#resting_blood_pressure (type: int): resting blood pressure

#chest_pain_type (type: int): chest pain type (4 values)

#num_major_vessels (type: int): number of major vessels (0-3) colored by flourosopy

#fasting_blood_sugar_gt_120_mg_per_dl (type: binary): fasting blood sugar > 120 mg/dl

#resting_ekg_results (type: int): resting electrocardiographic results (values 0,1,2)

#serum_cholesterol_mg_per_dl (type: int): serum cholestoral in mg/dl

#oldpeak_eq_st_depression (type: float): oldpeak = ST depression induced by exercise relative to rest, a measure of abnormality in electrocardiograms

#sex (type: binary): 0: female, 1: male

#age (type: int): age in years

#max_heart_rate_achieved (type: int): maximum heart rate achieved (beats per minute)

#exercise_induced_angina (type: binary): exercise-induced chest pain (0: False, 1: True)


# In[3]:


df = pd.read_csv('heart.csv')
df


# In[4]:


df.isnull().sum()


# In[5]:


print(df.info())


# In[6]:


#Findings the correlation

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap='terrain')

#We observe +ve correlation between target and cp, thalch, slop and also -ve correlation between target and sex, exang, ca, thai, oldpeak


# In[7]:


#To visualize the relationship between different features and figure out any linear relation between them we take help of PAIRPLOTS

sns.pairplot(data=df)


# In[8]:


#Box and Whiskers plot are useful to find out outliers in our data. If we have more outliers we will have to remove them or fix them; otherwise they will become as noise for the training data

df.plot(kind='box', subplots=True, layout=(5,3), figsize=(12,12))
plt.show()


# In[9]:


#With Histograms we can see the shape of each feature and provides the count of number of observations in each bin.

df.hist(figsize=(12,12), layout=(5,3))


# ## The features and their relation with the target( Heart Disease or No Heart Disease)

# In[10]:


sns.catplot(data=df, x='sex', y='age', hue='target', palette='husl')


# In[11]:


sns.barplot(data=df, x='sex', y='chol', hue='target', palette='spring')


# In[12]:


df['sex'].value_counts()


# In[13]:


#Chest pain type

df['cp'].value_counts()


# In[14]:


sns.countplot(data=df, x='cp', hue='target', palette='rocket')


# In[16]:


gen= pd.crosstab(df['sex'], df['target'])
print(gen)


# In[19]:


gen.plot(kind='bar', stacked=True, color=['blue','yellow'], grid=False)


# In[22]:


chest_pain= pd.crosstab(df['cp'],df['target'])

chest_pain.plot(kind='bar', stacked=True, color=['red','green'], grid=False)


# # Preparing Data for Model

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df [columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])


# # Data for Training

# In[25]:


x=df.drop(['target'], axis= 1)
y=df['target']


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3, random_state=40)


# In[28]:


#Checking sample size

print("x_train =", x_train.size)
print("x_test =", x_test.size)
print("y_train =", y_train.size)
print("y_train =", y_test.size)


# # Applying Logistic Regression Algorithm and finding the accuracy, precision and recall of the model

# In[41]:


#logistic regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression ()

model1= lr.fit(x_train, y_train)
prediction1 = model1.predict(x_test)


# In[43]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction1)
cm


# In[47]:


sns.heatmap(cm, annot=True, cmap="BuPu")


# In[48]:


TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print("Testing accuracy -", (TP+TN)/(TP+TN+FN+FP))


# In[49]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction1)


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction1))


# # Decision Tree

# In[53]:


from sklearn.tree import DecisionTreeClassifier

dtc= DecisionTreeClassifier()
model2=dtc.fit(x_train, y_train)
prediction2=model2.predict(x_test)
cm2= confusion_matrix(y_test, prediction2)

cm2


# In[54]:


accuracy_score(y_test, prediction2)


# In[55]:


print (classification_report(y_test, prediction2))


# # Random Forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier()
model3 = rfc.fit(x_train, y_train)
prediction3 = model3.predict(x_test)
cm3= confusion_matrix(y_test, prediction3)

cm3


# In[57]:


accuracy_score(y_test, prediction3)


# In[58]:


print (classification_report(y_test, prediction3))


# In[60]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[61]:


from sklearn.svm import SVC

svm = SVC()
model4= svm.fit(x_train, y_train)
prediction4=model4.predict(x_test)
cm4= confusion_matrix(y_test, prediction4)

cm4


# In[62]:


accuracy_score(y_test, prediction4)


# In[63]:


print (classification_report(y_test, prediction4))


# In[66]:


from sklearn.naive_bayes import GaussianNB

NB= GaussianNB()
model5=NB.fit(x_train,y_train)
prediction5= model5.predict(x_test)
cm5= confusion_matrix(y_test, prediction5)

cm5


# In[67]:


accuracy_score (y_test, prediction5)


# In[69]:


print (classification_report(y_test, prediction5))


# In[70]:


print("cm4", cm4)
print("-----------")
print("cm5", cm5)


# In[71]:


from sklearn.neighbors import KNeighborsClassifier

KNN= KNeighborsClassifier()
model6=KNN.fit(x_train, y_train)
prediction6= model6.predict(x_test)
cm6= confusion_matrix(y_test, prediction6)

cm6


# In[72]:


accuracy_score(y_test, prediction6)


# In[73]:


print(classification_report(y_test, prediction6))


# In[74]:


print('lr:', accuracy_score(y_test, prediction1))
print('dtc:', accuracy_score(y_test, prediction2))
print('rfc:', accuracy_score(y_test, prediction3))
print('NB:', accuracy_score(y_test, prediction4))
print('SVC:', accuracy_score(y_test, prediction5))
print('KNN:', accuracy_score(y_test, prediction6))


# # The Best Accuracy is Logistic Regression: 92

# In[ ]:




