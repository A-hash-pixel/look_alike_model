import os #provides functions for interacting with the operating system
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging
from utils import get_time

def predict(input_csv):
    """ takes a csv as input, predicts output and returns status of prediction, if prediction operation was success then it returns True, False otherwise"""
    try:
        df=pd.read_excel('{}'.format('./data/b2b.xlsx'))
        for column in df:
            unique_vals = np.unique(df[column])
            nr_values = len(unique_vals)
            if nr_values < 12:
                print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
            else:
                print('The number of values for feature {} :{}'.format(column, nr_values))
        new_raw_data = df
        new_raw_data['video'].max()
        X = new_raw_data.drop(['likely_customer'],axis='columns').values# Input features (attributes)
        y = new_raw_data['likely_customer'].values # Target vector
        print('X shape: {}'.format(np.shape(X)))
        print('y shape: {}'.format(np.shape(y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size=0.3, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
        rf.fit(X_train, y_train)
    
        df2=pd.read_excel("{}".format(input_csv))
        df3=df2[["video","cart_value","no_of_logins"]]
        result = rf.predict(df3)
        output = df2.copy()
        output['likely_customer'] = result
        if not output.empty:
            output_filename = './data/b2b_output.xlsx'
            output.to_excel(output_filename)
            os.remove(input_csv)
            return output_filename
        else:
            return None
    except:
        import traceback
        print(traceback.format_exc())
        return None

# predict("b2b.xlsx")
