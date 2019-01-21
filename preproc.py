
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers
from sklearn.linear_model import LinearRegression


# In[18]:


def drop_n(df, axis=0):
    """Drops rows or columns with NaNs"""
    if (axis == 0):
        return df.dropna(axis = 0)
    elif (axis == 1):
        return df.dropna(axis = 1)
    else:
        raise ValueError


# In[21]:


def fill_n(df, method = 'mean'):
    df = pd.DataFrame(df)
    for col in df.columns:
        if method == 'mean':
            measure = df[col].mean()
        elif method =='median':
            measure = df[col].median()
        elif method == 'mode':
            measure = df[col].mode()[0]
        else:
            raise ValueError
        df[col] = df[col].fillna(measure)
    return df


# In[15]:


def linear_pred(X, target):
    'NaNs inputted by linear regression'
    target_name = target.name
    data_dropped = drop_n(pd.concat([X, target], axis = 1), axis = 0)
    x_train, y_train = data_dropped[data_dropped.columns[:-1]], data_dropped[data_dropped.columns[-1]]
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    data_predict = pd.concat([fill_n(X,'mean'), target.fillna('m')], axis = 1)
    for i in range(len(data_predict)):
        if data_predict.loc[i, target_name] == 'm':
            x_test = pd.DataFrame(data_predict.iloc[i, :-1]).T
            data_predict.loc[i, target_name] = lr.predict(x_test)[0]
    return pd.to_numeric(data_predict[target_name])


# In[20]:


def knn(target, df, k_neighbors):
    n = len(target)
    # fill Nans with mean values
    df_kn = fill_n(df.iloc[:, :-1])
    # distance matrix
    distances = cdist(df_kn, df_kn, metric='euclidean')
    for i, v in enumerate(target):
        if pd.isnull(v):
            #order of vecotrs by distance by index
            order = distances[i].argsort()[:k_neighbors]
            # mean of neighbors
            target[i] = np.array([target[i] for i in order if not np.isnan(target[i])]).mean()
    return target


# In[5]:


def norm(df):
# normalization for numeric
    df = pd.DataFrame(df)
    for col in df.columns:
        df[col] = (df[col] - df[col].mean())/df[col].std()
    return df


# In[6]:


def scaling(df):
#scaling for numeric
        df = pd.DataFrame(df)
        for col in df.columns:
            df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
        return df

