#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis 

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# In[2]:


#Load the data
df = pd.read_csv('TRAJETORIAS_DATASET_Environmental_dimension_indicators.csv', quoting=3)
df.rename(columns=lambda x: x.replace('"', ''), inplace=True)


# In[3]:


df['state_abbrev'] = df['state_abbrev'].apply(lambda x: x.replace('"', ''))
df['tempp'] = df['tempp'].apply(lambda x: x.replace('"', ''))
df


# In[4]:


df = df.drop('state_abbrev', axis=1)
df = df.drop('pasture', axis=1)
df['tempp'] = df['tempp'].astype('float')

df.dtypes


# In[5]:


#Viewing the data 
df.head()


# In[6]:


df.describe()


# In[7]:


#Duplicate values
df.duplicated().sum()


# In[8]:


#Finding null values 
df.isnull().sum()


# In[9]:


#Finding unique values 
df['refor'].unique()
df['edge'].unique()
df['port'].unique()
df['river'].unique()
df['dgfor'].unique()
df['defor'].unique()


# In[10]:


sns.boxplot(df)


# In[11]:


# Delete rows using drop()
df.drop(df[df['edge'] >= 120].index, inplace = True)
print(df)


# In[12]:


from sklearn.preprocessing import MinMaxScaler

df=df.fillna(0)
# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Specify the columns to scale
columns_to_scale = ['geocode', 'period']

# Scale the selected columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Print the updated DataFrame
print(df)


# In[13]:


sns.boxplot(df)


# In[14]:


import math
df['edge'] = df['edge'].apply(lambda x: math.log(x+1))
df.drop(df[df['tempp'] >= 15].index, inplace = True)
df.drop(df[df['road'] >= 12].index, inplace = True)
sns.boxplot(df)
print(df)


# In[15]:


import seaborn as sns

sns.histplot(df['tempp'], kde=True)


# In[16]:


#Substituting non-skewed data column with mean
mean_value = df['refor'].mean()
df['refor'] = df['refor'].fillna(mean_value)

#Substituting skewed data columns with median
median_value = df[['edge','port','river','dgorg','dgfor','defor','deorg']].median()
df[['edge','port','river','dgorg','dgfor','defor','deorg']] = df[['edge','port','river','dgorg','dgfor','defor','deorg']].fillna(median_value)

df


# In[17]:


df.corr()


# In[18]:


#Correlation heatmap
sns.heatmap(df.corr())


# In[19]:


#Dropping columns with high correlation

df = df.drop('deorg', axis=1)
df = df.drop('dgorg', axis=1)
df = df.drop('core', axis=1)


# In[20]:


#Separating categorical and numerical variables 
cat_cols=df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)


# In[21]:


#Univariate analysis for categorical variables
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Bar plot for all categorical variables in the dataset')

sns.countplot(ax=axes[0, 0], x='state', data=df, color='blue',
              order=df['state'].value_counts().index)
sns.countplot(ax=axes[0, 1], x='municipality', data=df, color='blue',
              order=df['municipality'].value_counts().index)

axes[0][0].tick_params(labelrotation=45);
axes[0][1].tick_params(labelrotation=90);


# In[22]:


#Relationship between Categorical variables and continuous variables
fig, axarr = plt.subplots(4, 2, figsize=(12, 18))
df.groupby('refor')['fire'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12)
axarr[0][0].set_title("Refor vs Fire", fontsize=18)
df.groupby('refor')['tempp'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12)
axarr[0][1].set_title("Refor vs Tempp", fontsize=18)

plt.subplots_adjust(hspace=1.0)
plt.subplots_adjust(wspace=.5)
sns.despine()


# In[23]:


#Multivariate analysis
plt.figure(figsize=(12, 7))
#sns.heatmap(df.drop(['geocode','period','secveg','core','edge','port','dgfor','defor','precp','precn','tempp'],axis=1).corr(), annot = True, vmin = -1, vmax = 1)
sns.heatmap(df.drop(['geocode','period','secveg','edge'],axis=1).corr(), annot = True, vmin = -1, vmax = 1)
plt.show()


# In[24]:


features=df.drop(['geocode','period','secveg','edge','port','state','municipality','refor'],axis=1)
for i in features:
    sns.lmplot(x=i, y="refor", data=df,line_kws={'color': 'red'})
    text="Relation between refor and " + i 
    plt.title(text)
    plt.show()


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting a bar plot
plt.figure(figsize=(15, 7))
sns.barplot(x='state', y='refor', data=df)

plt.title("Average 'refor' by State")
plt.ylabel('refor')
plt.xlabel('State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()