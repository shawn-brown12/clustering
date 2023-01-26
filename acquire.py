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