import os #provides functions for interacting with the operating system
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from pywebio.input import *
from pywebio.output import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
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

        g = sns.pairplot(df, hue = 'likely_customer', diag_kws={'bw': 0.2})
        new_raw_data = df
        new_raw_data.shape
        new_raw_data['video'].max()
        X = new_raw_data.drop(['likely_customer'],axis='columns').values# Input features (attributes)
        y = new_raw_data['likely_customer'].values # Target vector
        print('X shape: {}'.format(np.shape(X)))
        print('y shape: {}'.format(np.shape(y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size=0.3, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
        rf.fit(X_train, y_train)
        prediction_test = rf.predict(X=X_test)
        # source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        # Accuracy on Test
        print("Training Accuracy is: {}".format(rf.score(X_train, y_train)))
        # Accuracy on Train
        print("Testing Accuracy is: {}".format(rf.score(X_test, y_test)))
        # Confusion Matrix
        #cm = confusion_matrix(y_test, prediction_test)
        #cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        #plt.figure()
        #plot_confusion_matrix(cm_norm, classes=rf.classes_)
        y_pred = rf.predict(X_train)
        # Plotting Confusion Matrix
        cm = confusion_matrix(y_train, y_pred)
        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plot_confusion_matrix(cm_norm, classes=rf.classes_, title='Training confusion')
        df2=pd.read_excel("{}".format(input_csv))
        df3=df2[["video","cart_value","no_of_logins"]]
        result = rf.predict(df3)
        output = df2.copy()
        output['likely_customer'] = result
        if not output.empty:
            output_filename = './data/b2b_output_{}.xlsx'.format(get_time())
            output.to_excel(output_filename)
            os.remove(input_csv)
            return output_filename
        else:
            return None
    except:
        import traceback
        print(traceback.format_exc())
        return None


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# predict("b2b.xlsx")
