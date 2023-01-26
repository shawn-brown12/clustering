import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from env import host, username, password

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

#----------------------------------------------------------    

def get_connection(db, user=username, host=host, password=password):
    '''
    This functions imports my credentials for the Codeup MySQL server to be used to pull data
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#----------------------------------------------------------    
    
def get_mall_customers():
    
    if os.path.isfile('mall_customers.csv'):
        
        df = pd.read_csv('mall_customers.csv')
        df = df.drop(columns='Unnamed: 0')

        return df

    else:
        
        url = get_connection('mall_customers')
        query = '''
                 SELECT *
                 FROM customers;
                 '''
        df = pd.read_sql(query, url)                
        df.to_csv('mall_customers.csv')

        return df
    
#----------------------------------------------------------    

def remove_outliers(df, k, col_list):
    ''' 
    This function takes in a dataframe, the threshold and a list of columns 
    and returns the dataframe with outliers removed
    '''   
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#----------------------------------------------------------    

def subset_df(df, stratify=None, seed=42):
    '''
    This function takes in a DataFrame and splits it into train, validate, test subsets for the modeling phase. Stratify is defaulted to None, but will take in a stratify if desired.
    '''
    train, val_test = train_test_split(df, train_size=.6, random_state=seed, stratify=stratify)
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=stratify)
    
    print(train.shape, validate.shape, test.shape)
    
    return train, validate, test

#----------------------------------------------------------  

def wrangle_mall_df(outlier_k=1.5):
    
    df = get_mall_customers()
    
    df = df.dropna()
    
    cont_list = ['age', 'annual_income', 'spending_score']
    
    cat_list = ['gender']

    df = remove_outliers(df, 1.5, cont_list)
    
    df = pd.get_dummies(df, cat_list)
    
    train, validate, test = subset_df(df)
    
    return train, validate, test

#----------------------------------------------------------    

def scale_data(train, validate, test, 
               scaler, columns_to_scale,
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so nothing gets messed up
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # make the scaler (unsure if redundant with addition I made)
    scaler = scaler
    # fit the scaler
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                             columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        
        return scaler, train_scaled, validate_scaled, test_scaled
    
    else:
        
        return train_scaled, validate_scaled, test_scaled

#----------------------------------------------------------    


#----------------------------------------------------------    


#----------------------------------------------------------    


#----------------------------------------------------------    
