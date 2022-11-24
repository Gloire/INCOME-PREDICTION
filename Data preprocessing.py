#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # For loading the data and processing it
import numpy as np
import seaborn as sns           # data visualization
import matplotlib.pyplot as plt # for data visualization

 
# Generate the charts just below the plot commands
get_ipython().run_line_magic('matplotlib', 'inline')


## import libraries for evaluating the templates
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
sns.set()


# define specific data types for each attribute
data_types =   {'hours.per.week'            : 'uint8',   
                'occupation'      : 'category',
                'fnlwgt'         : 'int32',   
                'sex'            : 'category',
                'education.num'  : 'uint8',   
                'relationship'   : 'category',
                'workclass'      : 'category',
                'marital.status'   : 'category',  
                'race'           : 'category',
                'education'      : 'category',
                'capital.loss'   : 'int32',   
                'capital.gain'   : 'int32',  
                'age'            : 'uint8',   
                'native.country' : 'category',
                'income'         : 'category'}

#load the data

df = pd.read_csv("file:///C:/Users/User/Desktop/adult.csv",
                 sep=',', na_values='?', dtype = data_types)
df.head()


# In[2]:




#display additional information about the dataset
df.info()


# In[7]:





df.rename({'income': 'target', 'education.num': 'education-num',
           'marital.status': 'marital-status', 'capital.gain': 'capital-gain',
           'capital.loss': 'capital-loss', 'hours.per.week': 'hours-per-week',
           'native.country': 'native-country'}, axis=1, inplace=True)
df.columns



# In[8]:




# Display summary for categorical attributes
df.describe(include='category')


# In[5]:




df.dtypes


# In[9]:


# display distribution of target attribute
plt.hist(df['sex'])
plt.title("Distribution of sex attribute", fontsize=20)
plt.show()


# In[10]:


#inspect the education attribute
df[['education-num', 'education']].value_counts().  reset_index().sort_values(by='education-num') 


# In[11]:


# Separate attributes into vectors according to data type (categorical or numeric)
cat_columns = list(df.drop(['target'], axis=1).select_dtypes(include=["category"]).columns)
print("cat_columns:", cat_columns, "=>", len(cat_columns))

num_columns = list(df.select_dtypes(include=["uint8", "int32", "int64", "float64"]).columns)
print("num_columns:", num_columns, "=>", len(num_columns))


# In[15]:


def count_plot(df, columns, label=None):
    plt.figure(figsize=(20, 12))
    for indx, var in enumerate(columns):
        plt.subplot(3, 3, indx + 1)
        if not label:
          g = sns.countplot(y=var, data=df,
                            order=df[var].value_counts().index)
        else:
          g = sns.countplot(y=var, data=df,
                            order=df[var].value_counts().index,
                            hue=label)
    plt.tight_layout()

def dist_plot(df, columns, type='boxplot', label=None):
    plt.figure(figsize=(20, 12))
    for indx, var in enumerate(columns):
        plt.subplot(3, 3, indx + 1)
        if (type=='boxplot'):
          if not label:
            g = sns.boxplot(x=var, data=df, showfliers=True)
          else:
            g = sns.boxplot(x=var, data=df, showfliers=True, y=label)
        elif (type=='histogram'):
          if not label:
            g = sns.histplot(x=var, data=df)
          else:
            g = sns.histplot(x=var, data=df, hue=label)
    plt.tight_layout()


# In[16]:




count_plot(df, cat_columns)


# In[17]:




# plot categorical columns (comparison based on sex)
count_plot(df, cat_columns, 'sex')


# In[18]:




dist_plot(df, num_columns)


# In[19]:


#plot numeric columns in a boxplot (comparison based on sex)
dist_plot(df, num_columns, label='target')


# In[20]:




# plot numeric columns(absolute quantities) in histograms
dist_plot(df, num_columns, type="histogram")


# In[21]:


dist_plot(df, num_columns, type="histogram", label='target')


# In[22]:




# analyse the general dataset
df.info()


# In[23]:


# Display quantities of missing values in each attribute
total = len(df)
for col in df.columns:
  qtde = len(df[df[col].isna()])
  if (qtde > 0):
    print('%-14s => %6d nulls (%2.1f%%)' % (col, qtde, qtde / total * 100))


# In[24]:


def analyze_outliers_iqr(X, features):

    indices = [x for x in X.index]
    qt_lines_total = len(indices)
    print('Number of lines:', qt_lines_total)
    
    print('Number of Attributes:', len(features))
    print('Attributes:', features)

    out_indexlist = []
    for col in features:
        Q1 = np.nanpercentile(X[col], 25.)
        Q2 = np.nanpercentile(X[col], 50.)
        Q3 = np.nanpercentile(X[col], 75.)
        
        cut_off = (Q3 - Q1) * 1.5
        upper, lower = Q3 + cut_off, Q1 - cut_off
        print('\nAttribute:', col)
        print('- Inferior limit: %.2f' % lower)
        print('- Median:         %.2f' % Q2)
        print('- Superior limit: %.2f' % upper)
                
        outliers_index = X[col][(X[col] < lower) | (X[col] > upper)].index.tolist()
        outliers = X[col][(X[col] < lower) | (X[col] > upper)].values
        qtd_outliers = len(outliers)
        print('- Number of outliers: %d (%.1f%%)' %               (qtd_outliers, qtd_outliers / qt_lines_total * 100))
        
        print('- Sample of outliers:', outliers[:50])
        
        out_indexlist.extend(outliers_index)
        
    out_indexlist = list(set(out_indexlist))
    out_indexlist.sort()
    qt_lines_outliers = len(out_indexlist)
    print('nNumber of lines with outliers: %d (%.1f%%)' %           (qt_lines_outliers, qt_lines_outliers / qt_lines_total * 100))


# In[25]:




# view report of outliers 
analyze_outliers_iqr(df, num_columns)


# In[26]:


#data preprocessing

before = len(df)
print("before: ", before, "lines")

# remove duplicate lines
df.drop_duplicates(inplace=True)

after = len(df)
print("After:", after, "lines")

print("\nRemove %d duplicate lines" % (before - after))


# In[27]:


# analyze column distribution (top 10 most frequent)
df['native-country'].value_counts(normalize=True).head(10)


# In[28]:




#remove column: 'native-country'
df.drop('native-country', axis=1, inplace=True)
cat_columns.remove('native-country')


# In[29]:


#remove education
df.drop('education', axis=1, inplace=True)
cat_columns.remove('education')


# In[30]:


#remove fnlwgt
df.drop('fnlwgt', axis=1, inplace=True)
num_columns.remove('fnlwgt')


# In[31]:


# display contents of categorical columns
for col in cat_columns:
  print("\n%s:" % col, df[col].unique())


# In[32]:


# grouping of the attribute "marital-status"
group_marital_status = {
  'Never-married': 'NotMarried',
  'Married-AF-spouse': 'Married',
  'Married-civ-spouse': 'Married',
  'Married-spouse-absent': 'NotMarried',
  'Separated': 'Separated',
  'Divorced': 'Separated',
  'Widowed': 'Widowed'
}
group_marital_status


# In[33]:


df.columns


# In[34]:


# group values of the attribute "marital-status"
df['marital-status-group'] = df['marital-status'].  map(group_marital_status).astype('category')
df['marital-status-group']


# In[35]:


#remove original column
df.drop('marital-status', axis=1, inplace=True)


# In[36]:


#visualize resulting distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='marital-status-group', data=df)
plt.show()


# In[37]:


# view general data frame
df.info()


# In[38]:


# update attribute vectors after dataset modifications
# separate attributes into vectors according to data type (categorical or numeric)
cat_columns = list(df.drop(['target'], axis=1).select_dtypes(include=["category"]).columns)
print("cat_columns:", cat_columns, "=>", len(cat_columns))

num_columns = list(df.select_dtypes(include='number').columns)
#num_columns = list(df.select_dtypes(include=["uint8", "int32", "int64", "float64"]).columns)
print("num_columns:", num_columns, "=>", len(num_columns))


# In[39]:


#display first 4 rows of the preprocessed dataset
df.head()


# In[40]:




# separate the dataset into two variables: the attributes/inputs (X) and the class/output (y)
X = df.drop(['target'], axis=1)
y = df['target'].values


# In[41]:




# substitute '<=50K' for 0, '>50K' for 1
y = np.array([0 if y=='<=50K' else 1 for y in y])
y


# In[ ]:




