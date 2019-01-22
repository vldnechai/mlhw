#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def dropna(df0, how = 'any', subset = np.nan, axis = 0):
    
    df = df0.reset_index(drop = True).copy()
    
    if subset == subset:
        df = df[subset]
        if len(subset) == 1:
            df = pd.DataFrame(df)
        
    if how == 'all':
        mode = any #leaves the row if at least one is non Nan
    else:
        mode = all #leaves the row if all are non Nan
        
    if axis in (0, 'index'):
        indexes = []
        for index, row in df.iterrows():
            if mode([obj == obj for obj in row]):
                indexes.append(index)
        df = df0.iloc[indexes,:]
        return df
    
    if axis in (1, 'columns'):
        columns = df0.columns
        for col in df:
            if mode([obj != obj for obj in df[col]]):
                columns.remove(col)
        df = df0[columns]
        return df
    
    raise SyntaxError('Invalid axis')


# In[3]:


def replacena(df0, subset = np.nan, mode = 'mean'):
    '''
    this function skips columns with inapplicable mode(behaviour)
    w/o raising and error
    '''
    df = df0.reset_index(drop = True).copy()
    
    descr = df.describe(include= 'all')
    
    if subset != subset:
        subset = df.columns
    
    #these keys come from describe dataframe indexing
    if mode == 'mean':
        key = 4
    elif mode == 'median':
        key = 8
    elif mode == 'mode':
        key = 3
    else:
        raise SyntaxError('Invalid mode')
    
    for col in subset:
        value = descr.iloc[key,col]
        if value != value:
            continue #skipping columns with inapplicable mode(behaviour)
        df[col] = [obj if obj == obj else value for obj in df[col]]
    
    return df


# In[4]:


from sklearn.linear_model import LinearRegression

def replacena_lr(df0, target, train = np.nan):
    '''
    target is column label
    if train is Nan, uses all entirely non-Nan columns in dataset
    '''
    
    df = df0.reset_index(drop = True).copy()
    
    descr = df.describe(include= 'all')
    key = 4 #mean value: exists only for numeric else Nan
    
    value = descr.iloc[key,target]
    if value != value:
        raise SyntaxError('Invalid target')
    
    if train != train:
        train = df.columns
        train = train.drop(target)
    
    #makes train_df with dummies of categorical variables(i.e. that have Nan mean in descr)
    train_df = pd.get_dummies(df[train], columns=[obj for obj in train if descr.iloc[key, obj] != descr.iloc[key, obj]])
    
    #append target to the end of train_df
    train_df = train_df.join(df.iloc[:,target])
    
    #get rows to predict
    pred_df = train_df[pd.isnull(train_df[target])]
    train_df = dropna(train_df)
    
    LR = LinearRegression()
    LR.fit(train_df.iloc[:,:-1], train_df.iloc[:,-1])
    
    pred = LR.predict(pred_df.iloc[:,:-1])
    
    i = 0
    for index, row in pred_df.iterrows():
        df.iloc[index, target] = pred[i]
        i += 1
    
    return df


# In[5]:


def eucl_distance(df0, p1, p2, train):
    return np.sum((df0.iloc[p1,train] - df0.iloc[p2,train])**2)


def KNN(df0, point_index, target, train, neighbours = 5):
   
    n = df0.shape[0]
    distance_matrix = np.zeros(n)
    
    for i in range (n):
        distance_matrix[i] = eucl_distance(df0, point_index, i, train)
        
    nearest_indexes = []
    while len(nearest_indexes) < neighbours:
        min_index = np.argmin(distance_matrix)
        if df0.iloc[min_index, target] != df0.iloc[min_index, target]:
            distance_matrix = np.delete(distance_matrix, min_index)
        else:
            nearest_indexes.append(min_index)
            distance_matrix = np.delete(distance_matrix, min_index)
    return np.mean(df0.iloc[nearest_indexes, target])


# In[6]:


def replacenaKNN(df0, target, train = np.nan, neighbours = 5):
    
    df = df0.reset_index(drop = True).copy()
    
    descr = df.describe(include= 'all')
    key = 4 #mean value: exists only for numeric else Nan
    
    value = descr.iloc[key,target]
    if value != value:
        raise SyntaxError('Invalid target')
    
    if train != train:
        train = df.columns
        
    try:
        for j in range(df.shape[0]):
            if df.iloc[j, target] != df.iloc[j, target]:
                df.iloc[j, target] = KNN(df, j, target, train, neighbours = neighbours)
    except IndexError:
        return df
    return df


# In[7]:


def normalize(df0, columns):
    
    df = df0.copy()
    
    descr = df.describe(include= 'all')
    
    for col in columns:
        mean = descr.iloc[4, col] #mean value: exists only for numeric else Nan
        if mean != mean:
            continue
        std = descr.iloc[5, col] #standart deviation
        df.iloc[:,col] = [(obj - mean)/ std for obj in df.iloc[:,col]]
    return df


# In[8]:


def standardize(df0, columns):
    
    df = df0.copy()
    
    descr = df.describe(include= 'all')
    
    for col in columns:
        minimum = descr.iloc[6, col] #minimum value: exists only for numeric else Nan
        if minimum != minimum:
            continue
        maximum = descr.iloc[10, col] #maximum value
        try:
            df.iloc[:,col] = [(obj - minimum)/ (maximum - minimum) for obj in df.iloc[:,col]]
        except ZeroDivisionError:
            continue
    return df

