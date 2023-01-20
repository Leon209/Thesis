import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder, StandardScaler

from scipy.io.arff import loadarff 

from livelossplot import PlotLosses
from collections import Counter

import os
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from IPython.display import Image
from IPython.display import display, clear_output

import pandas as pd

import warnings
warnings.filterwarnings('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#os.environ["PYTHONWARNINGS"] = "default"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

import tensorflow as tf
import tensorflow_addons as tfa

from skmultiflow.data import SEAGenerator 
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import RandomRBFGeneratorDrift

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
#tf.get_logger().setLevel('WARNING')
#tf.autograph.set_verbosity(1)

np.seterr(all="ignore")

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


import seaborn as sns
sns.set_style("darkgrid")

import time
import random

from utilities.utilities_GDT import *
from utilities.GDT_for_streams import *

from joblib import Parallel, delayed

from itertools import product
from collections.abc import Iterable
import collections

from copy import deepcopy
import timeit

from xgboost import XGBClassifier
#from genetic_tree import GeneticTree

from keras.utils import np_utils
import scipy

from pathlib import Path
import csv
import dill

from tabulate import tabulate


def flatten_list(l):
    
    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
                
    flat_l = flatten(l)
    
    return list(flat_l)

def flatten_dict(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def mergeDict(dict1, dict2):
    #Merge dictionaries and keep values of common keys in list
    newDict = {**dict1, **dict2}
    for key, value in newDict.items():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                newDict[key] = mergeDict(dict1[key], value)
            elif isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend(value)
            elif isinstance(dict1[key], list) and not isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend([value])
            elif not isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = [dict1[key]]
                newDict[key].extend(value)
            else:
                newDict[key] = [dict1[key], value]
    return newDict


def normalize_data(X_data,normalizer_list, technique='min-max', low=-1, high=1):
    if isinstance(X_data, pd.DataFrame):
        if not normalizer_list:
            for column_name in X_data:
                if technique == 'min-max':
                    scaler = MinMaxScaler(feature_range=(low, high))
                elif technique == 'mean':
                    scaler = StandardScaler()
                scaler.fit(X_data[column_name].values.reshape(-1, 1))
                X_data[column_name] = scaler.transform(X_data[column_name].values.reshape(-1, 1)).ravel()
                normalizer_list.append(scaler)
        else:
            i=0
            for column_name in X_data:
                X_data[column_name] = normalizer_list[i].transform(X_data[column_name].values.reshape(-1, 1)).ravel()
                i = i+1
    else:
        if not normalizer_list:
            for i, column in enumerate(X_data.T):
                if technique == 'min-max':
                    scaler = MinMaxScaler(feature_range=(low, high))
                elif technique == 'mean':
                    scaler = StandardScaler()       
                scaler.fit(column.reshape(-1, 1))
                X_data[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
                normalizer_list.append(scaler)
        else:
            for i, column in enumerate(X_data.T):     
                X_data[:,i] = normalizer_list[i].transform(column.reshape(-1, 1)).ravel()
        
    return X_data, normalizer_list

def split_train_test_valid(X_data, y_data, valid_frac=0.20, test_frac=0.20, seed=42, verbosity=0):
    data_size = X_data.shape[0]
    test_size = int(data_size*test_frac)
    valid_size = int(data_size*valid_frac)
    
    X_train_with_valid, X_test, y_train_with_valid, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_with_valid, y_train_with_valid, test_size=valid_size, random_state=seed)

    if verbosity > 0:
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)
        print(X_test.shape, y_test.shape)    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_with_valid, y_train_with_valid


def rebalance_data(X_train, 
                   y_train, 
                   balance_ratio=0.25, 
                   strategy='SMOTE',#'SMOTE', 
                   seed=42, 
                   verbosity=0):#, strategy='SMOTE'
    
    
    min_label = min(Counter(y_train).values())
    sum_label = sum(Counter(y_train).values())

    min_ratio = min_label/sum_label
    if verbosity > 0:
        print('Min Ratio: ', str(min_ratio))    
    if min_ratio <= balance_ratio/(len(Counter(y_train).values()) - 1):
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, SMOTENC
        from imblearn.combine import SMOTETomek, SMOTEENN
        try:
            if strategy == 'SMOTE':
                oversample = SMOTE()
            elif strategy == 'SMOTEN':
                oversample = SMOTEN()                 
            elif strategy == 'BorderlineSMOTE':
                oversample = BorderlineSMOTE()                
            elif strategy == 'KMeansSMOTE':
                oversample = KMeansSMOTE(cluster_balance_threshold=0.1)    
            elif strategy == 'SVMSMOTE':
                oversample = SVMSMOTE()   
            elif strategy == 'SMOTETomek':
                oversample = SMOTETomek()   
            elif strategy == 'SMOTEENN':
                oversample = SMOTEENN()               
            elif strategy == 'ADASYN':
                oversample = ADASYN()
            else:
                oversample = RandomOverSampler(sampling_strategy='auto', random_state=seed)

            X_train, y_train = oversample.fit_resample(X_train, y_train)
        except ValueError:
            oversample = RandomOverSampler(sampling_strategy='auto', random_state=seed)
            X_train, y_train = oversample.fit_resample(X_train, y_train)
            
        min_label = min(Counter(y_train).values())
        sum_label = sum(Counter(y_train).values())
        min_ratio = min_label/sum_label
        if verbosity > 0:
            print('Min Ratio: ', str(min_ratio))    

    return X_train, y_train





def rebalance_data_old(X_train, 
                   y_train, 
                   balance_ratio=0.25, 
                   strategy='SMOTE', 
                   seed=42, 
                   verbosity=0):#, strategy='SMOTE'
    
    if max(y_train) == 1:
        true_labels = len(y_train[y_train >= 0.5 ]) 
        false_labels = len(y_train[y_train < 0.5 ]) 

        true_ratio = true_labels/(true_labels+false_labels)
        false_ratio = false_labels/(false_labels+true_labels)

        min_ratio = min(true_ratio, false_ratio)
        if verbosity > 0:
            print('True Ratio: ', str(true_labels/(true_labels+false_labels)))    
        if min_ratio <= balance_ratio:
            from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, SMOTENC
            from imblearn.combine import SMOTETomek, SMOTEENN
            if strategy == 'SMOTE':
                oversample = SMOTE()
            elif strategy == 'SMOTEN':
                oversample = SMOTEN()                 
            elif strategy == 'BorderlineSMOTE':
                oversample = BorderlineSMOTE()                
            elif strategy == 'KMeansSMOTE':
                oversample = KMeansSMOTE(cluster_balance_threshold=0.1)    
            elif strategy == 'SVMSMOTE':
                oversample = SVMSMOTE()   
            elif strategy == 'SMOTETomek':
                oversample = SMOTETomek()   
            elif strategy == 'SMOTEENN':
                oversample = SMOTEENN()               
            elif strategy == 'ADASYN':
                oversample = ADASYN()
            else:
                oversample = RandomOverSampler(sampling_strategy='auto', random_state=seed)

            X_train, y_train = oversample.fit_resample(X_train, y_train)

            true_labels = len(y_train[y_train >= 0.5 ]) 
            false_labels = len(y_train[y_train < 0.5 ]) 
            if verbosity > 0:
                print('True Ratio: ', str(true_labels/(true_labels+false_labels)))   

    return X_train, y_train





def preprocess_data(X_data, 
                    y_data,
                    nominal_features,
                    ordinal_features,
                    config,
                    normalizer_list,
                    random_seed=42,
                    verbosity=0):

    start_evaluate_network_complete = time.time()

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    
    if verbosity > 0:
        print('Original Data Shape (selected): ', X_data.shape)

    X_data = X_data.astype(np.float64)

    if verbosity > 0:
        print('Original Data Shape (encoded): ', X_data.shape)
        if config['gdt']['objective'] == 'classification':
            print('Original Data Class Distribution: ', y_data[y_data>=0.5].shape[0], ' (true) /', y_data[y_data<0.5].shape[0], ' (false)')

    
    if config['preprocessing']['normalization_technique'] is not None:
        X_data, normalizer_list = normalize_data(X_data, normalizer_list, technique=config['preprocessing']['normalization_technique'])
    else:
        normalizer_list = None
    
    (X_train, 
     y_train, 
     X_valid, 
     y_valid, 
     X_test, 
     y_test,
     X_train_with_valid,
     y_train_with_valid) = split_train_test_valid(X_data, 
                                      y_data, 
                                      seed=random_seed,
                                      verbosity=verbosity)  
        
    if config['gdt']['objective'] == 'classification':
        X_train, y_train = rebalance_data(X_train, 
                                          y_train, 
                                          balance_ratio=config['preprocessing']['balance_threshold'],  
                                          strategy='SMOTE',
                                          verbosity=verbosity)
        
        #
        X_train_with_valid, y_train_with_valid = rebalance_data(X_train_with_valid, 
                                          y_train_with_valid, 
                                          balance_ratio=config['preprocessing']['balance_threshold'],  
                                          strategy='SMOTE',
                                          verbosity=verbosity)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), (X_train_with_valid, y_train_with_valid), normalizer_list







def load_dataset_for_streams(identifier, 
                             max_total_samples,
                             random_seed=42, 
                             config=None,
                             verbosity=0):
    
    if identifier == 'BIN:rbf_f':
        feature_names = [
                        'att1', #numeric
                        'att2', #numeric
                        'att3', #numeric
                        'att4', #numeric
                        'att5', #numeric
                        'att6', #numeric
                        'att7', #numeric     
                        'att8', #numeric
                        'att9', #numeric
                        'att10',#numeric
                        'class' #binary
                        ]
        
        data = pd.read_csv('./datasets_streaming/rbf_f.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    if identifier == 'BIN:rbf_m':
        feature_names = [
                        'att1', #numeric
                        'att2', #numeric
                        'att3', #numeric
                        'att4', #numeric
                        'att5', #numeric
                        'att6', #numeric
                        'att7', #numeric     
                        'att8', #numeric
                        'att9', #numeric
                        'att10',#numeric
                        'class' #binary
                        ]
        
        data = pd.read_csv('./datasets_streaming/rbf_m.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    if identifier == 'BIN:hyperplane':
        feature_names = [
                        'att1', #numeric
                        'att2', #numeric
                        'att3', #numeric
                        'att4', #numeric
                        'att5', #numeric
                        'att6', #numeric
                        'att7', #numeric     
                        'att8', #numeric
                        'att9', #numeric
                        'att10',#numeric
                        'class' #binary
                        ]
        
        stream = HyperplaneGenerator(mag_change=0.001)

        temp = stream.next_sample(500000)
        stacked_data = np.column_stack((temp[0],temp[1]))
        data = pd.DataFrame(data = stacked_data, columns = feature_names)
        
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    
    if identifier == 'BIN:airlines':
        feature_names = [
                        'airline', #nominal
                        'flight', #numeric
                        'airport_from', #nominal
                        'airport_to', #nominal
                        'day_of_week', #nominal
                        'time',#numeric
                        'length',#numeric
                        'class'#binary
                        ]
        
        data = pd.read_csv('./datasets_streaming/airlines.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)
        
        features_select = [
                        'airline', #nominal
                        'flight', #numeric
                        'airport_from', #nominal
                        'airport_to', #nominal
                        'day_of_week', #nominal
                        'time',#numeric
                        'length',#numeric
                        'class'#binary
                        ]
        
        data = data[features_select]

        nominal_features = ['airline',
                           'day_of_week']
        ordinal_features = ['airport_from',
                           'airport_to']

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    
    
    if identifier == 'BIN:agr_a':
        feature_names = [
                        'salary', #numeric
                        'commission', #numeric
                        'age', #numeric
                        'elevel', #ordinal
                        'car', #numeric
                        'zipcode',#nominal
                        'hvalue',
                        'hyears',      
                        'loan', #numeric
                        'class' #binary
                        ]
        
        data = pd.read_csv('./datasets_streaming/agr_a.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)
        
        features_select = [
                        'salary', #numeric
                        'commission', #numeric
                        'age',#numeric
                        'elevel',
                        'car', #numeric  
                        'zipcode',
                        'hvalue',
                        'hyears',  
                        'loan', #numeric
                        'class', #binary
                        ]
        
        data = data[features_select]

        nominal_features = ['zipcode',
                           'car']
        ordinal_features = ['elevel']

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    
    if identifier == 'BIN:agr_g':
        feature_names = [
                        'salary', #numeric
                        'commission', #numeric
                        'age', #numeric
                        'elevel', #ordinal
                        'car', #numeric
                        'zipcode',#nominal
                        'hvalue',
                        'hyears',      
                        'loan', #numeric
                        'class' #binary
                        ]
        
        data = pd.read_csv('./datasets_streaming/agr_g.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)
        
        features_select = [
                        'salary', #numeric
                        'commission', #numeric
                        'age',#numeric
                        'elevel',
                        'car', #nominal  
                        'zipcode',
                        'hvalue',
                        'hyears',  
                        'loan', #numeric
                        'class', #binary
                        ]
        
        data = data[features_select]

        nominal_features = ['zipcode',
                           'car'
                            ]
        ordinal_features = ['elevel']

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    if identifier == 'BIN:sea_a':
        feature_names = [
                        'attribute1', #numeric   
                        'attribute2', #numeric  
                        'attribute3', #numeric  
                        'class', #binary  
                        ]
        
        data = pd.read_csv('./datasets_streaming/sea_a.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)
        
        features_select = [
                        'attribute1', #numeric   
                        'attribute2', #numeric  
                        'attribute3', #numeric  
                        'class', #binary  
                        ]
        
        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    
    if identifier == 'BIN:sea_g':
        feature_names = [
                        'attribute1', #numeric   
                        'attribute2', #numeric  
                        'attribute3', #numeric  
                        'class', #binary  
                        ]
        
        data = pd.read_csv('./datasets_streaming/sea_g.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        if(len(data) > max_total_samples):
            data = data.head(max_total_samples)
        
        features_select = [
                        'attribute1', #numeric   
                        'attribute2', #numeric  
                        'attribute3', #numeric  
                        'class', #binary  
                        ]
        
        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
        
    if identifier == 'BIN:electricity':
        feature_names = [
                        'date', #ignore.
                        'day', #nominal maybe use later
                        'period', #numeric
                        'nswprice', #ignore
                        'nswdemand', #numeric
                        'vicprice',#numeric
                        'vicdemand',#numeric
                        'transfer' #numeric
                        ]
        class_name = ['class']
        
        X_data = pd.read_csv('./datasets_streaming/elec2_data.csv', names=feature_names, index_col=False, delimiter=' ', header=0)
        y_data = pd.read_csv('./datasets_streaming/elec2_label.csv', names=class_name, index_col=False, delimiter=' ', header=0)
        
        if(len(X_data) > max_total_samples):
            X_data = X_data.head(max_total_samples)
            y_data = y_data.head(max_total_samples)
        
        features_select = [
                        'period', #numeric
                        'nswdemand', #numeric
                        'vicprice',#numeric
                        'vicdemand',#numeric
                        'transfer', #numeric
                        'day'
                        ]
        
        X_data = X_data[features_select]

        nominal_features = ['day']
        ordinal_features = []

        y_data = pd.Series(OrdinalEncoder().fit_transform(y_data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
    
    
        
    if identifier == 'BIN:NOAA_Weather':
        feature_names = [
                       'Temperature', #numeric
                       'Dew Point', #numeric
                       'Sea Level Pressure', #numeric
                       'Visibility', #numeric
                       'Average Wind Speed', #numeric
                       'Maximum Sustained Wind Speed', #numeric
                       'Maximum Temperature', #numeric
                       'Minimum Temperature', #numeric
                        ]
        
        class_name = ['class']
        
        X_data = pd.read_csv('./datasets_streaming/NEweather_data.csv', names=feature_names, index_col=False, delimiter=',', header=0)
        y_data = pd.read_csv('./datasets_streaming/NEweather_class.csv', names=class_name, index_col=False, delimiter=',', header=0)
        

        if(len(X_data) > max_total_samples):
            X_data = X_data.head(max_total_samples)
            y_data = y_data.head(max_total_samples)
        
        features_select = [
                       'Temperature', #numeric
                       'Dew Point', #numeric
                       'Sea Level Pressure', #numeric
                       'Visibility', #numeric
                       'Average Wind Speed', #numeric
                       'Maximum Sustained Wind Speed', #numeric
                       'Maximum Temperature', #numeric
                       'Minimum Temperature', #numeric
                        ]
        
        X_data = X_data[features_select]

        nominal_features = []
        ordinal_features = []

        y_data = pd.Series(OrdinalEncoder().fit_transform(y_data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        return encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features)
        
    
    
def encode_ordinal_and_nominal_features(X_data, y_data, nominal_features, ordinal_features):
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), nominal_features)], remainder='passthrough', sparse_threshold=0)
    transformer.fit(X_data)
    X_data = transformer.transform(X_data)
    X_data = pd.DataFrame(X_data, columns=transformer.get_feature_names())
    
    
    for ordinal_feature in ordinal_features:
        X_data[ordinal_feature] = OrdinalEncoder().fit_transform(X_data[ordinal_feature].values.reshape(-1, 1)).flatten()
    
    return X_data, y_data, nominal_features, ordinal_features
    
    
    

def get_preprocessed_dataset(identifier, 
                             random_seed=42, 
                             config=None,
                             verbosity=0):
        
    if identifier == 'BIN:Cervical Cancer':
        data = pd.read_csv('./real_world_datasets/Cervical_Cancer/risk_factors_cervical_cancer.csv', index_col=False)#, names=feature_names

        features_select = [
                            'Age', #numeric
                            'Number of sexual partners', #numeric
                            'First sexual intercourse', #numeric
                            'Num of pregnancies', #numeric
                            'Smokes', #binary
                            'Smokes (years)', #numeric
                            'Hormonal Contraceptives', #binary
                            'Hormonal Contraceptives (years)', #numeric
                            'IUD', #binary
                            'IUD (years)', #numeric
                            'STDs', #binary
                            'STDs (number)', #numeric
                            'STDs: Number of diagnosis', #numeric
                            'STDs: Time since first diagnosis', #numeric
                            'STDs: Time since last diagnosis', #numeric
                            'Biopsy'
                           ]

        data = data[features_select]

        data['Number of sexual partners'][data['Number of sexual partners'] == '?'] = data['Number of sexual partners'].mode()[0]
        data['First sexual intercourse'][data['First sexual intercourse'] == '?'] = data['First sexual intercourse'].mode()[0]
        data['Num of pregnancies'][data['Num of pregnancies'] == '?'] = data['Num of pregnancies'].mode()[0]
        data['Smokes'][data['Smokes'] == '?'] = data['Smokes'].mode()[0]
        data['Smokes (years)'][data['Smokes (years)'] == '?'] = data['Smokes (years)'].mode()[0]
        data['Hormonal Contraceptives'][data['Hormonal Contraceptives'] == '?'] = data['Hormonal Contraceptives'].mode()[0]
        data['Hormonal Contraceptives (years)'][data['Hormonal Contraceptives (years)'] == '?'] = data['Hormonal Contraceptives (years)'].mode()[0]
        data['IUD'][data['IUD'] == '?'] = data['IUD'].mode()[0]
        data['IUD (years)'][data['IUD (years)'] == '?'] = data['IUD (years)'].mode()[0]
        data['STDs'][data['STDs'] == '?'] = data['STDs'].mode()[0]
        data['STDs (number)'][data['STDs (number)'] == '?'] = data['STDs (number)'].mode()[0]
        data['STDs: Time since first diagnosis'][data['STDs: Time since first diagnosis'] == '?'] = data['STDs: Time since first diagnosis'][data['STDs: Time since first diagnosis'] != '?'].mode()[0]
        data['STDs: Time since last diagnosis'][data['STDs: Time since last diagnosis'] == '?'] = data['STDs: Time since last diagnosis'][data['STDs: Time since last diagnosis'] != '?'].mode()[0]

        nominal_features = [
                            ]
        
        ordinal_features = [
                            ]


        X_data = data.drop(['Biopsy'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Biopsy'].values.reshape(-1, 1)).flatten(), name='Biopsy')

    elif identifier == 'BIN:Credit Card':
        data = pd.read_csv('./real_world_datasets/UCI_Credit_Card/UCI_Credit_Card.csv', index_col=False)
        data = data.drop(['ID'], axis = 1)

        features = ['LIMIT_BAL', #numeric
                     'SEX', #binary
                     'EDUCATION', #categorical
                     'MARRIAGE', #binary
                     'AGE', #numeric
                     'PAY_0', #categorical
                     'PAY_2', #categorical
                     'PAY_3', #categorical
                     'PAY_4', #categorical
                     'PAY_5', #categorical
                     'PAY_6', #categorical
                     'BILL_AMT1',  #numeric
                     'BILL_AMT2', #numeric
                     'BILL_AMT3', #numeric
                     'BILL_AMT4', #numeric
                     'BILL_AMT5', #numeric
                     'BILL_AMT6', #numeric
                     'PAY_AMT1', #numeric
                     'PAY_AMT2', #numeric
                     'PAY_AMT3', #numeric
                     'PAY_AMT4', #numeric
                     'PAY_AMT5', #numeric
                     'PAY_AMT6']   #numeric      
        
        nominal_features = [
                            ]

        ordinal_features = [
                            ]

        X_data = data.drop(['default.payment.next.month'], axis = 1)
        y_data = ((data['default.payment.next.month'] < 1) * 1)
      
    elif identifier == 'BIN:Absenteeism':

        data = pd.read_csv('real_world_datasets/Absenteeism/absenteeism.csv', delimiter=';')

        features_select = [
                                   'Seasons', #nominal
                                   'Day of the week',#nominal
                                   'Month of absence', #nominal
                                   'Disciplinary failure', #binary
                                   'Social drinker', #binary
                                   'Social smoker', #binary
                                   'Transportation expense', #numeric
                                   'Distance from Residence to Work', #numeric
                                   'Service time',  #numeric
                                   'Age', #numeric
                                   'Work load Average/day ', #numeric 
                                   'Hit target', #numeric
                                   'Education', #categorical
                                   'Son', #numeric 
                                   'Pet', #numeric 
                                   'Weight', #numeric 
                                   'Height', #numeric 
                                   'Body mass index', #numeric 
                                   'Absenteeism time in hours'
                                ]

        data = data[features_select]

        nominal_features = [
                            ]

        ordinal_features = [
                            ]

        X_data = data.drop(['Absenteeism time in hours'], axis = 1)
        y_data = ((data['Absenteeism time in hours'] > 4) * 1) #absenteeism_data['Absenteeism time in hours']

    elif identifier == 'BIN:Adult':
        feature_names = [
                         "Age", #0 numeric
                         "Workclass",  #1 nominal
                         "fnlwgt",  #2 numeric
                         "Education",  #3 nominal
                         "Education-Num",  #4 nominal
                         "Marital Status", #5 nominal
                         "Occupation",  #6 nominal
                         "Relationship",  #7 nominal
                         "Race",  #8 nominal
                         "Sex",  #9 binary
                         "Capital Gain",  #10 numeric
                         "Capital Loss", #11 numeric
                         "Hours per week",  #12 numeric
                         "Country", #13 nominal
                         "capital_gain" #14
                        ] 

        data = pd.read_csv('./real_world_datasets/Adult/adult.data', names=feature_names, index_col=False)


        #adult_data['Workclass'][adult_data['Workclass'] != ' Private'] = 'Other'
        #adult_data['Race'][adult_data['Race'] != ' White'] = 'Other'

        #adult_data.head()

        features_select = [
                         "Sex",  #9 
                         "Race",  #8
                         "Workclass",  #1
                         "Age", #0
                         "fnlwgt",  #2
                         "Education",  #3
                         "Education-Num",  #4
                         "Marital Status", #5
                         "Occupation",  #6
                         "Relationship",  #7
                         "Capital Gain",  #10
                         "Capital Loss", #11
                         "Hours per week",  #12
                         "Country", #13 
                         "capital_gain"
                          ]

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                              'Race', 
                              'Workclass', 
                              'Education',
                              "Marital Status",
                              "Occupation", 
                              "Relationship", 
                              "Country",
                              'Sex',
                           ]

        X_data = data.drop(['capital_gain'], axis = 1)
        y_data = ((data['capital_gain'] != ' <=50K') * 1)        
        
    elif identifier == 'BIN:Titanic':
        data = pd.read_csv("./real_world_datasets/Titanic/train.csv")

        data['Age'].fillna(data['Age'].mean(), inplace = True)
        data['Fare'].fillna(data['Fare'].mean(), inplace = True)

        data['Embarked'].fillna('S', inplace = True)

        features_select = [
                            #'Cabin', 
                            #'Ticket', 
                            #'Name', 
                            #'PassengerId'    
                            'Sex', #binary
                            'Embarked', #nominal
                            'Pclass', #nominal
                            'Age', #numeric
                            'SibSp', #numeric 
                            'Parch', #numeric
                            'Fare', #numeric 
                            'Survived',    
                          ]

        data = data[features_select]

        nominal_features = []#[1, 2, 7]
        ordinal_features = ['Embarked',
                            'Sex']

        X_data = data.drop(['Survived'], axis = 1)
        y_data = data['Survived']

    elif identifier == 'BIN:Loan House':
        data = pd.read_csv('real_world_datasets/Loan/loan-train.csv', delimiter=',')

        data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
        data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
        data['Married'].fillna(data['Married'].mode()[0], inplace=True)
        data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
        data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
        data['Credit_History'].fillna(data['Credit_History'].mean(), inplace=True)

        features_select = [
                            #'Loan_ID', 
                            'Gender', #binary
                            'Married',  #binary
                            'Dependents', #numeric
                            'Education', # #binary
                            'Self_Employed',  #binary
                            'ApplicantIncome', #numeric
                            'CoapplicantIncome', #numeric
                            'LoanAmount', #numeric
                            'Loan_Amount_Term', #numeric
                            'Credit_History', #binary
                            'Property_Area', #nominal
                            'Loan_Status'
                            ]

        data = data[features_select]

        #loan_data['Dependents'][loan_data['Dependents'] == '3+'] = 4
        #loan_data['Dependents'] = loan_data['Dependents'].astype(int)

        #loan_data['Property_Area'][loan_data['Property_Area'] == 'Rural'] = 0
        #loan_data['Property_Area'][loan_data['Property_Area'] == 'Semiurban'] = 1
        #loan_data['Property_Area'][loan_data['Property_Area'] == 'Urban'] = 2
        #loan_data['Property_Area'] = loan_data['Property_Area'].astype(int)

        nominal_features = [
   
                                ]


        ordinal_features = [
                            'Dependents',
                            'Property_Area',             
                            'Education',
                            'Gender', 
                            'Married', 
                            'Self_Employed',
                           ]

        X_data = data.drop(['Loan_Status'], axis = 1)
        y_data = ((data['Loan_Status'] == 'Y') * 1)         
        
    elif identifier == 'BIN:Loan Credit':

        data = pd.read_csv('real_world_datasets/Credit Loan/train_split.csv', delimiter=',')

        data['emp_title'].fillna(data['emp_title'].mode()[0], inplace=True)
        data['emp_length'].fillna(data['emp_length'].mode()[0], inplace=True)
        #data['desc'].fillna(data['desc'].mode()[0], inplace=True)
        data['title'].fillna(data['title'].mode()[0], inplace=True)
        #data['mths_since_last_delinq'].fillna(data['mths_since_last_delinq'].mode()[0], inplace=True)
        #data['mths_since_last_record'].fillna(data['mths_since_last_record'].mode()[0], inplace=True)
        data['revol_util'].fillna(data['revol_util'].mode()[0], inplace=True)
        data['collections_12_mths_ex_med'].fillna(data['collections_12_mths_ex_med'].mode()[0], inplace=True)
        #data['mths_since_last_major_derog'].fillna(data['mths_since_last_major_derog'].mode()[0], inplace=True)
        #data['verification_status_joint'].fillna(data['verification_status_joint'].mode()[0], inplace=True)
        data['tot_coll_amt'].fillna(data['tot_coll_amt'].mode()[0], inplace=True)
        data['tot_cur_bal'].fillna(data['tot_cur_bal'].mode()[0], inplace=True)
        data['total_rev_hi_lim'].fillna(data['total_rev_hi_lim'].mode()[0], inplace=True)


        ##remove too many null
        #'mths_since_last_delinq','mths_since_last_record', 'mths_since_last_major_derog','pymnt_plan','desc', 'verification_status_joint'


        features_select = [
                            #'member_id', 
                            'loan_amnt', 
                            'funded_amnt', 
                            'funded_amnt_inv', 
                            'term',
                            #'batch_enrolled',
                            'int_rate', 
                            'grade', 
                            #'sub_grade', 
                            #'emp_title',
                            'emp_length',
                            'home_ownership', 
                            'annual_inc', 
                            'verification_status',
                            #'pymnt_plan', 
                            #'desc', 
                            'purpose', 
                            #'title', 
                            #'zip_code', 
                            #'addr_state',
                            'dti', 
                            'delinq_2yrs', 
                            'inq_last_6mths', 
                            #'mths_since_last_delinq',
                            #'mths_since_last_record',
                            'open_acc', 
                            'pub_rec', 
                            'revol_bal',
                            'revol_util', 
                            'total_acc', 
                            'initial_list_status', 
                            'total_rec_int',
                            'total_rec_late_fee', 
                            'recoveries', 
                            'collection_recovery_fee',
                            'collections_12_mths_ex_med', 
                            #'mths_since_last_major_derog',
                            'application_type', 
                            #'verification_status_joint', 
                            'last_week_pay',
                            'acc_now_delinq', 
                            'tot_coll_amt', 
                            'tot_cur_bal', 
                            'total_rev_hi_lim',
                            'loan_status'
                            ]

        data = data[features_select]

        nominal_features = [

                                ]
        ordinal_features = [
                            #'member_id', 
                            'loan_amnt', 
                            'funded_amnt', 
                            'funded_amnt_inv', 
                            'term',
                            #'batch_enrolled',
                            'int_rate', 
                            'grade', 
                            #'sub_grade', 
                            #'emp_title',
                            'emp_length',
                            'home_ownership', 
                            'annual_inc', 
                            'verification_status',
                            #'pymnt_plan', 
                            #'desc', 
                            'purpose', 
                            #'title', 
                            #'zip_code', 
                            #'addr_state',
                            'dti', 
                            'delinq_2yrs', 
                            'inq_last_6mths', 
                            #'mths_since_last_delinq',
                            #'mths_since_last_record',
                            'open_acc', 
                            'pub_rec', 
                            'revol_bal',
                            'revol_util', 
                            'total_acc', 
                            'initial_list_status', 
                            'total_rec_int',
                            'total_rec_late_fee', 
                            'recoveries', 
                            'collection_recovery_fee',
                            'collections_12_mths_ex_med', 
                            #'mths_since_last_major_derog',
                            'application_type', 
                            #'verification_status_joint', 
                            'last_week_pay',
                            'acc_now_delinq', 
                            'tot_coll_amt', 
                            'tot_cur_bal', 
                            'total_rev_hi_lim',
                           ]

        X_data = data.drop(['loan_status'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['loan_status'].values.reshape(-1, 1)).flatten(), name='loan_status')

    elif identifier == 'BIN:Medical Insurance':
        
        data = pd.read_csv('real_world_datasets/Medical Insurance/insurance.csv', delimiter=',')

        features_select = [
                            'age', 
                            'sex', 
                            'bmi', 
                            'children', 
                            'smoker',
                            'region',
                            'charges'
                            ]

        data = data[features_select]

        nominal_features = [
                            ]
        ordinal_features = [
                            'region',
                            'sex',
                            'smoker'
                           ]


        X_data = data.drop(['charges'], axis = 1)
        y_data = ((data['charges'] > 10_000) * 1)

    elif identifier == 'BIN:Bank Marketing':

        data = pd.read_csv('real_world_datasets/Bank Marketing/bank-full.csv', delimiter=';') #bank

        features_select = [
                            'age', #numeric
                            'job', #nominal
                            'marital', #nominal
                            'education', #nominal
                            'default', #nominal
                            'housing', #nominal
                            'loan', #nominal
                            'contact', #binary
                            'month', #nominal
                            'day', #nominal
                            #'duration', #numeric
                            'campaign', #nominal
                            'pdays', #numeric
                            'previous', #numeric
                            'poutcome', #nominal
                            'y',
                            ]

        data = data[features_select]

        nominal_features = [

                                ]
        ordinal_features = [
                            'job', #nominal
                            'marital', #nominal
                            'education', #nominal
                            'default', #nominal
                            'housing', #nominal
                            'loan', #nominal
                            'contact', #binary
                            'month', #nominal
                            'day', #nominal
                            'campaign', #nominal
                            'pdays', #numeric
                            'poutcome', #nominal            
                           ]


        X_data = data.drop(['y'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['y'].values.reshape(-1, 1)).flatten(), name='y')

    elif identifier == 'BIN:Wisconsin Breast Cancer Original':
        
        feature_names = [
                        'Sample code number',
                        'Clump Thickness',
                        'Uniformity of Cell Size',
                        'Uniformity of Cell Shape',
                        'Marginal Adhesion',
                        'Single Epithelial Cell Size',
                        'Bare Nuclei',
                        'Bland Chromatin',
                        'Normal Nucleoli',
                        'Mitoses',
                        'Class',
                        ]

        data = pd.read_csv('./real_world_datasets/Wisconsin Breast Cancer Original/breast-cancer-wisconsin.data', names=feature_names, index_col=False)

        data['Clump Thickness'][data['Clump Thickness'] == '?'] = data['Clump Thickness'].mode()[0]
        data['Uniformity of Cell Size'][data['Uniformity of Cell Size'] == '?'] = data['Uniformity of Cell Size'].mode()[0]
        data['Uniformity of Cell Shape'][data['Uniformity of Cell Shape'] == '?'] = data['Uniformity of Cell Shape'].mode()[0]
        data['Marginal Adhesion'][data['Marginal Adhesion'] == '?'] = data['Marginal Adhesion'].mode()[0]
        data['Single Epithelial Cell Size'][data['Single Epithelial Cell Size'] == '?'] = data['Single Epithelial Cell Size'].mode()[0]
        data['Bare Nuclei'][data['Bare Nuclei'] == '?'] = data['Bare Nuclei'].mode()[0]
        data['Bland Chromatin'][data['Bland Chromatin'] == '?'] = data['Bland Chromatin'].mode()[0]
        data['Normal Nucleoli'][data['Normal Nucleoli'] == '?'] = data['Normal Nucleoli'].mode()[0]
        data['Mitoses'][data['Mitoses'] == '?'] = data['Mitoses'].mode()[0]

        features_select = [
                        #'Sample code number',
                        'Clump Thickness',
                        'Uniformity of Cell Size',
                        'Uniformity of Cell Shape',
                        'Marginal Adhesion',
                        'Single Epithelial Cell Size',
                        'Bare Nuclei',
                        'Bland Chromatin',
                        'Normal Nucleoli',
                        'Mitoses',
                        'Class',
                            ]

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                           ]


        X_data = data.drop(['Class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class'].values.reshape(-1, 1)).flatten(), name='Class')

    elif identifier == 'BIN:Wisconsin Diagnostic Breast Cancer':

        feature_names = [
                        'ID number',
                        'Diagnosis',
                        'radius',# (mean of distances from center to points on the perimeter)
                        'texture',# (standard deviation of gray-scale values)
                        'perimeter',
                        'area',
                        'smoothness',# (local variation in radius lengths)
                        'compactness',# (perimeter^2 / area - 1.0)
                        'concavity',# (severity of concave portions of the contour)
                        'concave points',# (number of concave portions of the contour)
                        'symmetry',
                        'fractal dimension',# ("coastline approximation" - 1)
                        ]
        #Wisconsin Diagnostic Breast Cancer
        data = pd.read_csv('./real_world_datasets/Wisconsin Diagnostic Breast Cancer/wdbc.data', names=feature_names, index_col=False)

        features_select = [
                            #'ID number',
                            'Diagnosis',
                            'radius',# numeric
                            'texture',# numeric
                            'perimeter',# numeric
                            'area',# numeric
                            'smoothness',# numeric
                            'compactness',# numeric 
                            'concavity',# numeric
                            'concave points',# numeric
                            'symmetry',# numeric
                            'fractal dimension',# numeric
                            ]

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                           ]


        X_data = data.drop(['Diagnosis'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Diagnosis'].values.reshape(-1, 1)).flatten(), name='Diagnosis')
       
    elif identifier == 'BIN:Wisconsin Prognostic Breast Cancer':

        feature_names = [
                        'ID number',
                        'Diagnosis',
                        'radius',# (mean of distances from center to points on the perimeter)
                        'texture',# (standard deviation of gray-scale values)
                        'perimeter',
                        'area',
                        'smoothness',# (local variation in radius lengths)
                        'compactness',# (perimeter^2 / area - 1.0)
                        'concavity',# (severity of concave portions of the contour)
                        'concave points',# (number of concave portions of the contour)
                        'symmetry',
                        'fractal dimension',# ("coastline approximation" - 1)
                        ]
        #Wisconsin Prognostic Breast Cancer
        data = pd.read_csv('./real_world_datasets/Wisconsin Prognostic Breast Cancer/wpbc.data', names=feature_names, index_col=False)

        features_select = [
                            #'ID number',
                            'Diagnosis',
                            'radius',# (mean of distances from center to points on the perimeter)
                            'texture',# (standard deviation of gray-scale values)
                            'perimeter',
                            'area',
                            'smoothness',# (local variation in radius lengths)
                            'compactness',# (perimeter^2 / area - 1.0)
                            'concavity',# (severity of concave portions of the contour)
                            'concave points',# (number of concave portions of the contour)
                            'symmetry',
                            'fractal dimension',# ("coastline approximation" - 1)
                            ]

        data = data[features_select]

        nominal_features = [
                            ]
        ordinal_features = [
                            ]

        X_data = data.drop(['Diagnosis'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Diagnosis'].values.reshape(-1, 1)).flatten(), name='Diagnosis')

    elif identifier == 'MULT:Abalone':
        
        feature_names = [
                        'Sex',#		nominal			M, F, and I (infant)
                        'Length',#	continuous	mm	Longest shell measurement
                        'Diameter',#	continuous	mm	perpendicular to length
                        'Height',#		continuous	mm	with meat in shell
                        'Whole weight',#	continuous	grams	whole abalone
                        'Shucked weight',#	continuous	grams	weight of meat
                        'Viscera weight',#	continuous	grams	gut weight (after bleeding)
                        'Shell weight',#	continuous	grams	after being dried
                        'Rings',#		integer			+1.5 gives the age in years
                        ]

        data = pd.read_csv('./real_world_datasets/Abalone/abalone.data', names=feature_names, index_col=False)


        features_select = [
                        'Sex',#		nominal			M, F, and I (infant)
                        'Length',#	continuous	mm	Longest shell measurement
                        'Diameter',#	continuous	mm	perpendicular to length
                        'Height',#		continuous	mm	with meat in shell
                        'Whole weight',#	continuous	grams	whole abalone
                        'Shucked weight',#	continuous	grams	weight of meat
                        'Viscera weight',#	continuous	grams	gut weight (after bleeding)
                        'Shell weight',#	continuous	grams	after being dried
                        'Rings',#		integer			+1.5 gives the age in years
                            ]

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                                'Sex',
                           ]

        X_data = data.drop(['Rings'], axis = 1)
        y_data = data['Rings']

    elif identifier == 'MULT:Car':
        feature_names = [
           'buying',#       v-high, high, med, low
           'maint',#        v-high, high, med, low
           'doors',#        2, 3, 4, 5-more
           'persons',#      2, 4, more
           'lug_boot',#     small, med, big
           'safety',#       low, med, high
           'class',#        unacc, acc, good, v-good
                        ]

        data = pd.read_csv('./real_world_datasets/Car/car.data', names=feature_names, index_col=False)

        features_select = [
                           'buying',#       v-high, high, med, low
                           'maint',#        v-high, high, med, low
                           'doors',#        2, 3, 4, 5-more
                           'persons',#      2, 4, more
                           'lug_boot',#     small, med, big
                           'safety',#       low, med, high
                           'class',#        unacc, acc, good, v-good
                            ]

        data = data[features_select]

        nominal_features = [
                                ]

        ordinal_features = [
                               'buying',#       v-high, high, med, low
                               'maint',#        v-high, high, med, low
                               'doors',#        2, 3, 4, 5-more
                               'persons',#      2, 4, more
                               'lug_boot',#     small, med, big
                               'safety',#       low, med, high            
                           ]



        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class') #((data['class'] != 'unacc') * 1)        

    elif identifier == 'BIN:Heart Disease':
        feature_names = [
           'age',# numeric    
           'sex',# binary   
           'cp',# nominal
           'trestbps',# numeric
           'chol',# numeric   
           'fbs',# binary    
           'restecg',# nominal
           'thalach',# numeric  
           'exang',# binary 
           'oldpeak',# numeric     
           'slope',# nominal
           'ca',# numeric
           'thal',# nominal  
           'num',#     
                        ]

        data = pd.read_csv('./real_world_datasets/Heart Disease/processed.cleveland.data', names=feature_names, index_col=False) #, delimiter=' '

        nominal_features = [
                                ]

        ordinal_features = [
                           ]


        data['age'][data['age'] == '?'] = data['age'].mode()[0]
        data['sex'][data['sex'] == '?'] = data['sex'].mode()[0]
        data['cp'][data['cp'] == '?'] = data['cp'].mode()[0]
        data['trestbps'][data['trestbps'] == '?'] = data['trestbps'].mode()[0]
        data['chol'][data['chol'] == '?'] = data['chol'].mode()[0]
        data['fbs'][data['fbs'] == '?'] = data['fbs'].mode()[0]
        data['restecg'][data['restecg'] == '?'] = data['restecg'].mode()[0]
        data['thalach'][data['thalach'] == '?'] = data['thalach'].mode()[0]
        data['exang'][data['exang'] == '?'] = data['exang'].mode()[0]
        data['oldpeak'][data['oldpeak'] == '?'] = data['oldpeak'].mode()[0]
        data['slope'][data['slope'] == '?'] = data['slope'].mode()[0]
        data['ca'][data['ca'] == '?'] = data['ca'].mode()[0]
        data['thal'][data['thal'] == '?'] = data['thal'].mode()[0]

        X_data = data.drop(['num'], axis = 1)
        y_data = ((data['num'] < 1) * 1)
        
    elif identifier == 'BIN:Habermans Survival':

        feature_names = [
           'age',#      
           'year',#   
           'nodes_detected',#      
           'survival',#     
                        ]

        data = pd.read_csv('./real_world_datasets/Habermans/haberman.data', names=feature_names, index_col=False) #, delimiter=' '


        nominal_features = [
                                ]

        ordinal_features = [
                           ]


        X_data = data.drop(['survival'], axis = 1)
        y_data = ((data['survival'] < 2) * 1)

    elif identifier == 'BIN:Heart Failure':
        
        data = pd.read_csv('real_world_datasets/Heart Failure/heart_failure_clinical_records_dataset.csv', delimiter=',')

        features = [
                'age',# continuous #### age of the patient (years) 
                'anaemia',# binary #### decrease of red blood cells or hemoglobin (boolean)
                'high blood pressure',# binary #### if the patient has hypertension (boolean)
                'creatinine phosphokinase (CPK)',# continuous #### level of the CPK enzyme in the blood (mcg/L)
                'diabetes',# binary #### if the patient has diabetes (boolean)
                'ejection fraction',# continuous #### percentage of blood leaving the heart at each contraction (percentage)
                'platelets',# continuous #### platelets in the blood (kiloplatelets/mL)
                'sex',# binary ####  woman or man (binary)
                'serum creatinine',# continuous #### level of serum creatinine in the blood (mg/dL)
                'serum sodium',# continuous #### level of serum sodium in the blood (mEq/L)
                'smoking',# binary #### if the patient smokes or not (boolean)
                'time',# continuous #### follow-up period (days)
                'target'# death event: if the patient deceased during the follow-up period (boolean)        

        ]

        nominal_features = [
                                ]
        ordinal_features = [

                           ]


        X_data = data.drop(['DEATH_EVENT'], axis = 1)
        y_data = ((data['DEATH_EVENT'] > 0) * 1)
     
    elif identifier == 'MULT:Iris':
        
        feature_names = [
                        'sepal_length',
                        'sepal_width',
                        'petal_length',
                        'petal_width',
                        'class',
                        ]
        #Wisconsin Prognostic Breast Cancer
        data = pd.read_csv('./real_world_datasets/Iris/iris.data', names=feature_names, index_col=False)
        
        features_select = [
                        'sepal_length',
                        'sepal_width',
                        'petal_length',
                        'petal_width',
                        'class',
                            ]

        data = data[features_select]

        nominal_features = [
                            ]
        ordinal_features = [
                            ]
        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')    
    
    elif identifier == 'MULT:Annealing':
        
        feature_names = [
            'family',# --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS
            'product-type',# C, H, G
            'steel',# -,R,A,U,K,M,S,W,V
            'carbon',#: continuous
            'hardness',#: continuous
            'temper_rolling',#: -,T
            'condition',#: -,S,A,X
            'formability',#: -,1,2,3,4,5
            'strength',#: continuous
            'non-ageing',#: -,N
            'surface-finish',#: P,M,-
            'surface-quality',#: -,D,E,F,G
            'enamelability',#: -,1,2,3,4,5
            'bc',#: Y,-
            'bf',#: Y,-
            'bt',#: Y,-
            'bw/me',#: B,M,-
            'bl',#: Y,-
            'm',#: Y,-
            'chrom',#: C,-
            'phos',#: P,-
            'cbond',#: Y,-
            'marvi',#: Y,-
            'exptl',#: Y,-
            'ferro',#: Y,-
            'corr',#: Y,-
            'blue/bright/varn/clean',#: B,R,V,C,-
            'lustre',#: Y,-
            'jurofm',#: Y,-
            's',#: Y,-
            'p',#: Y,-
            'shape',#: COIL, SHEET
            'thick',#: continuous
            'width',#: continuous
            'len',#: continuous
            'oil',#: -,Y,N
            'bore',#: 0000,0500,0600,0760
            'packing',#: -,1,2,3
            'classes',#: 1,2,3,4,5,U
                        ]

        data = pd.read_csv('./real_world_datasets/Annealing/anneal.data', names=feature_names, index_col=False)

        features_select = [
                            'family',# --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS
                            'product-type',# C, H, G
                            'steel',# -,R,A,U,K,M,S,W,V
                            'carbon',#: continuous
                            'hardness',#: continuous
                            'temper_rolling',#: -,T
                            'condition',#: -,S,A,X
                            'formability',#: -,1,2,3,4,5
                            'strength',#: continuous
                            'non-ageing',#: -,N
                            'surface-finish',#: P,M,-
                            'surface-quality',#: -,D,E,F,G
                            'enamelability',#: -,1,2,3,4,5
                            'bc',#: Y,-
                            'bf',#: Y,-
                            'bt',#: Y,-
                            'bw/me',#: B,M,-
                            'bl',#: Y,-
                            'm',#: Y,-
                            'chrom',#: C,-
                            'phos',#: P,-
                            'cbond',#: Y,-
                            'marvi',#: Y,-
                            'exptl',#: Y,-
                            'ferro',#: Y,-
                            'corr',#: Y,-
                            'blue/bright/varn/clean',#: B,R,V,C,-
                            'lustre',#: Y,-
                            'jurofm',#: Y,-
                            's',#: Y,-
                            'p',#: Y,-
                            'shape',#: COIL, SHEET
                            'thick',#: continuous
                            'width',#: continuous
                            'len',#: continuous
                            'oil',#: -,Y,N
                            'bore',#: 0000,0500,0600,0760
                            'packing',#: -,1,2,3
                            'classes',#: 1,2,3,4,5,U
                            ]

        data = data[features_select]

        nominal_features = [

                            ]
        ordinal_features = [
                            'family',# --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS
                            'product-type',# C, H, G
                            'steel',# -,R,A,U,K,M,S,W,V
                            'temper_rolling',#: -,T
                            'condition',#: -,S,A,X         
                            'bw/me',#: B,M,-
                            'blue/bright/varn/clean',#: B,R,V,C,-
                            'oil',#: -,Y,N
                            'bore',#: 0000,0500,0600,0760
            
            
            
            

                            'formability',#: -,1,2,3,4,5
                            'non-ageing',#: -,N
                            'surface-finish',#: P,M,-
                            'surface-quality',#: -,D,E,F,G
                            'enamelability',#: -,1,2,3,4,5
                            'bc',#: Y,-
                            'bf',#: Y,-
                            'bt',#: Y,-
                            'bl',#: Y,-
                            'm',#: Y,-
                            'chrom',#: C,-
                            'phos',#: P,-
                            'cbond',#: Y,-
                            'marvi',#: Y,-
                            'exptl',#: Y,-
                            'ferro',#: Y,-
                            'corr',#: Y,-
                            'lustre',#: Y,-
                            'jurofm',#: Y,-
                            's',#: Y,-
                            'p',#: Y,-
                            'shape',#: COIL, SHEET
                            'packing',#: -,1,2,3
                            ]

        X_data = data.drop(['classes'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['classes'].values.reshape(-1, 1)).flatten(), name='classes')          
    
    elif identifier == 'MULT:Glass':
    
        feature_names = [
                        'Id number',#: 1 to 214
                        'RI',#: refractive index
                        'Na',#: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
                        'Mg',#: Magnesium
                        'Al',#: Aluminum
                        'Si',#: Silicon
                        'K',#: Potassium
                        'Ca',#: Calcium
                        'Ba',#: Barium
                        'Fe',#: Iron
                        'Type of glass',#: (class attribute)
                        ]

        data = pd.read_csv('./real_world_datasets/Glass/glass.data', names=feature_names, index_col=False)

        features_select = [
                            #'Id number',#: 1 to 214
                            'RI',#: refractive index
                            'Na',#: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
                            'Mg',#: Magnesium
                            'Al',#: Aluminum
                            'Si',#: Silicon
                            'K',#: Potassium
                            'Ca',#: Calcium
                            'Ba',#: Barium
                            'Fe',#: Iron
                            'Type of glass',#: (class attribute)
                            ]

        data = data[features_select]

        nominal_features = [
                            ]
        ordinal_features = [
                            ]

        X_data = data.drop(['Type of glass'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Type of glass'].values.reshape(-1, 1)).flatten(), name='Type of glass')       
    
    elif identifier == 'MULT:Solar Flare':
        
        feature_names = [
                            'Code for class',# (modified Zurich class) (A,B,C,D,E,F,H)
                            'Code for largest spot size',# (X,R,S,A,H,K)
                            'Code for spot distribution',# (X,O,I,C)
                            'Activity',# (1 = reduced, 2 = unchanged)
                            'Evolution',# (1 = decay, 2 = no growth, 3 = growth)
                            'Previous 24 hour flare activity code',# (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
                            'Historically-complex',# (1 = Yes, 2 = No)
                            'Did region become historically complex on this pass across the suns disk',# (1 = yes, 2 = no)
                            'Area',# (1 = small, 2 = large)
                            'Area of the largest spot',# (1 = <=5, 2 = >5)

                            'C-class flares production by this region in the following 24 hours (common flares)',#; Number
                            'M-class flares production by this region in the following 24 hours (moderate flares)',#; Number
                            'X-class flares production by this region in the following 24 hours (severe flares)',#; Number
                        ]

        data1 = pd.read_csv('./real_world_datasets/Solar Flare/flare.data1', names=feature_names, index_col=False, delimiter=' ', header=0)
        data2 = pd.read_csv('./real_world_datasets/Solar Flare/flare.data2', names=feature_names, index_col=False, delimiter=' ', header=0)
        
        data = pd.concat([data1, data2])
        
        features_select = [
                                'Code for class',# (modified Zurich class) (A,B,C,D,E,F,H)
                                'Code for largest spot size',# (X,R,S,A,H,K)
                                'Code for spot distribution',# (X,O,I,C)
                                'Activity',# (1 = reduced, 2 = unchanged)
                                'Evolution',# (1 = decay, 2 = no growth, 3 = growth)
                                'Previous 24 hour flare activity code',# (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
                                'Historically-complex',# (1 = Yes, 2 = No)
                                'Did region become historically complex on this pass across the suns disk',# (1 = yes, 2 = no)
                                'Area',# (1 = small, 2 = large)
                                'Area of the largest spot',# (1 = <=5, 2 = >5)

                                'C-class flares production by this region in the following 24 hours (common flares)',#; Number
                                #'M-class flares production by this region in the following 24 hours (moderate flares)',#; Number
                                #'X-class flares production by this region in the following 24 hours (severe flares)',#; Number
                            ]

        data = data[features_select]

        nominal_features = [

                            ]
        ordinal_features = [
                                'Code for class',# (modified Zurich class) (A,B,C,D,E,F,H)
                                'Code for largest spot size',# (X,R,S,A,H,K)
                                'Code for spot distribution',# (X,O,I,C)      
            
            
            
            
            
                                'Activity',# (1 = reduced, 2 = unchanged)
                                'Evolution',# (1 = decay, 2 = no growth, 3 = growth)
                                'Previous 24 hour flare activity code',# (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
                                'Historically-complex',# (1 = Yes, 2 = No)
                                'Did region become historically complex on this pass across the suns disk',# (1 = yes, 2 = no)
                                'Area',# (1 = small, 2 = large)
                                'Area of the largest spot',# (1 = <=5, 2 = >5)        
                            ]

        X_data = data.drop(['C-class flares production by this region in the following 24 hours (common flares)'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['C-class flares production by this region in the following 24 hours (common flares)'].values.reshape(-1, 1)).flatten(), name='C-class flares production by this region in the following 24 hours (common flares)')    
    
    elif identifier == 'MULT:Splice':
        
        feature_names = flatten_list([
                            'One of {n ei ie}',#, indicating the class.
                            'instance name',
                            'sequence',
                        ])

        data_raw = pd.read_csv('./real_world_datasets/Splice/splice.data', names=feature_names, index_col=False)#, header=0)
        data_np = np.hstack([data_raw['One of {n ei ie}'].values.reshape(-1,1), data_raw['instance name'].values.reshape(-1,1), np.array([split_string[-61:-1] for split_string in data_raw['sequence'].str.split('')])])
        
        columnnames = flatten_list([
                            'One of {n ei ie}',#, indicating the class.
                            'instance name',
                            [str(i) for i in range(-30,30)],
                        ])
        data = pd.DataFrame(data=data_np, columns=columnnames)
        
        features_select = flatten_list([
                            'One of {n ei ie}',#, indicating the class.
                            #'instance name',
                            [str(i) for i in range(-30,30)],
                        ])

        data = data[features_select]

        nominal_features = flatten_list([
                        ])
        ordinal_features = flatten_list([   
                            [str(i) for i in range(-30,30)],
                            ])

        X_data = data.drop(['One of {n ei ie}'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['One of {n ei ie}'].values.reshape(-1, 1)).flatten(), name='One of {n ei ie}')
                               
    elif identifier == 'MULT:Wine':
        
        feature_names = [
                            'Alcohol',
                            'Malic acid',
                            'Ash',
                            'Alcalinity of ash',
                            'Magnesium',
                            'Total phenols',
                            'Flavanoids',
                            'Nonflavanoid phenols',
                            'Proanthocyanins',
                            'Color intensity',
                            'Hue',
                            'OD280/OD315 of diluted wines',
                            'Proline',
                        ]

        data = pd.read_csv('./real_world_datasets/Wine/wine.data', names=feature_names, index_col=False)#, header=0)
        
        features_select = [
                            'Alcohol',
                            'Malic acid',
                            'Ash',
                            'Alcalinity of ash',
                            'Magnesium',
                            'Total phenols',
                            'Flavanoids',
                            'Nonflavanoid phenols',
                            'Proanthocyanins',
                            'Color intensity',
                            'Hue',
                            'OD280/OD315 of diluted wines',
                            'Proline',
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                            ]

        X_data = data.drop(['Alcohol'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Alcohol'].values.reshape(-1, 1)).flatten(), name='Alcohol')
                           
    elif identifier == 'MULT:Dermatology':

        feature_names = [
                        'erythema',#
                        'scaling',#
                        'definite borders',#
                        'itching',#
                        'koebner phenomenon',#
                        'polygonal papules',#
                        'follicular papules',#
                        'oral mucosal involvement',#
                        'knee and elbow involvement',#
                        'scalp involvement',#
                        'family history',#, (0 or 1)
                        'melanin incontinence',#
                        'eosinophils in the infiltrate',#
                        'PNL infiltrate',#
                        'fibrosis of the papillary dermis',#
                        'exocytosis',#
                        'acanthosis',#
                        'hyperkeratosis',#
                        'parakeratosis',#
                        'clubbing of the rete ridges',#
                        'elongation of the rete ridges',#
                        'thinning of the suprapapillary epidermis',#
                        'spongiform pustule',#
                        'munro microabcess',#
                        'focal hypergranulosis',#
                        'disappearance of the granular layer',#
                        'vacuolisation and damage of basal layer',#
                        'spongiosis',#
                        'saw-tooth appearance of retes',#
                        'follicular horn plug',#
                        'perifollicular parakeratosis',#
                        'inflammatory monoluclear inflitrate',#
                        'band-like infiltrate',#
                        'Age (linear)',#
                        'diagnosis'
                        ]

        data = pd.read_csv('./real_world_datasets/Dermatology/dermatology.data', names=feature_names, index_col=False)
        #data['Age (linear)'].fillna(data['Age (linear)'].mean(), inplace = True)
        data['Age (linear)'].replace(['?'], pd.to_numeric(data['Age (linear)'], errors='coerce').mean(), inplace = True)
        data['Age (linear)'] = data['Age (linear)'].astype(float)
        features_select = [
                        'erythema',#
                        'scaling',#
                        'definite borders',#
                        'itching',#
                        'koebner phenomenon',#
                        'polygonal papules',#
                        'follicular papules',#
                        'oral mucosal involvement',#
                        'knee and elbow involvement',#
                        'scalp involvement',#
                        'family history',#, (0 or 1)
                        'melanin incontinence',#
                        'eosinophils in the infiltrate',#
                        'PNL infiltrate',#
                        'fibrosis of the papillary dermis',#
                        'exocytosis',#
                        'acanthosis',#
                        'hyperkeratosis',#
                        'parakeratosis',#
                        'clubbing of the rete ridges',#
                        'elongation of the rete ridges',#
                        'thinning of the suprapapillary epidermis',#
                        'spongiform pustule',#
                        'munro microabcess',#
                        'focal hypergranulosis',#
                        'disappearance of the granular layer',#
                        'vacuolisation and damage of basal layer',#
                        'spongiosis',#
                        'saw-tooth appearance of retes',#
                        'follicular horn plug',#
                        'perifollicular parakeratosis',#
                        'inflammatory monoluclear inflitrate',#
                        'band-like infiltrate',#
                        'Age (linear)',#
                        'diagnosis'
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                            ]

        X_data = data.drop(['diagnosis'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['diagnosis'].values.reshape(-1, 1)).flatten(), name='diagnosis')            
            
    elif identifier == 'MULT:Balance Scale':

        feature_names = [
                        'Class Name',#: 3 (L, B, R)
                        'Left-Weight',#: 5 (1, 2, 3, 4, 5)
                        'Left-Distance',#: 5 (1, 2, 3, 4, 5)
                        'Right-Weight',#: 5 (1, 2, 3, 4, 5)
                        'Right-Distance',#: 5 (1, 2, 3, 4, 5)
                        ]

        data = pd.read_csv('./real_world_datasets/Balance Scale/balance-scale.data', names=feature_names, index_col=False)#, header=0)

        features_select = [
                        'Class Name',#: 3 (L, B, R)
                        'Left-Weight',#: 5 (1, 2, 3, 4, 5)
                        'Left-Distance',#: 5 (1, 2, 3, 4, 5)
                        'Right-Weight',#: 5 (1, 2, 3, 4, 5)
                        'Right-Distance',#: 5 (1, 2, 3, 4, 5)
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                            ]

        X_data = data.drop(['Class Name'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class Name'].values.reshape(-1, 1)).flatten(), name='Class Name')            
            
    elif identifier == 'BIN:Blood Transfusion':

        feature_names = [
                        'R',# (Recency - months since last donation),
                        'F',# (Frequency - total number of donation),
                        'M',# (Monetary - total blood donated in c.c.),
                        'T',# (Time - months since first donation), and
                        'a',# binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).
                        ]

        data = pd.read_csv('./real_world_datasets/Transfusion/transfusion.data', names=feature_names, index_col=False, header=0)#, header=0)

        features_select = [
                        'R',# (Recency - months since last donation),
                        'F',# (Frequency - total number of donation),
                        'M',# (Monetary - total blood donated in c.c.),
                        'T',# (Time - months since first donation), and
                        'a',# binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                            ]

        X_data = data.drop(['a'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['a'].values.reshape(-1, 1)).flatten(), name='a') 
        
    elif identifier == 'BIN:agr_a':
        feature_names = [
                        'salary', #numeric
                        'commission', #numeric
                        'age', #numeric
                        'elevel', #ka
                        'car', #numeric
                        'zipcode',
                        'hvalue',
                        'hyears',      
                        'loan', #numeric
                        'class' #binary
                        ]
        
        data = pd.read_csv('./datasets_streaming/agr_a.csv', names=feature_names, index_col=False, delimiter=',', header=0)#.head(50000)
        print(data)
        
        features_select = [
                        'salary', #numeric
                        'commission', #numeric
                        'age', #numeric
                        'car', #numeric   
                        'loan', #numeric
                        'class' #binary
                        ]
        
        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')
        
        
            
    elif identifier == 'BIN:German':

        feature_names = [
                        'Status of existing checking account', #nominal
                        'Duration in month', #numeric
                        'Credit history', #nominal
                        'Purpose', #nominal
                        'Credit amount', #numeric
                        'Savings account/bonds', #nominal
                        'Present employment since', #nominal
                        'Installment rate in percentage of disposable income',#numeric
                        'Personal status and sex', #nominal
                        'Other debtors / guarantors', #nominal
                        'Present residence since', #numeric
                        'Property', #nominal
                        'Age in years', #numeric
                        'Other installment plans', #nominal
                        'Housing', #nominal
                        'Number of existing credits at this bank', #numeric
                        'Job', #nominal
                        'Number of people being liable to provide maintenance for', #numeric
                        'Telephone', #binary
                        'foreign worker', #binary
                        'label'
                        ]            
 

        data = pd.read_csv('./real_world_datasets/German/german.data', names=feature_names, index_col=False, delimiter=' ')#, header=0)#, header=0)

        features_select = [
                        'Status of existing checking account',
                        'Duration in month',
                        'Credit history',
                        'Purpose',
                        'Credit amount',
                        'Savings account/bonds',
                        'Present employment since',
                        'Installment rate in percentage of disposable income',
                        'Personal status and sex',
                        'Other debtors / guarantors',
                        'Present residence since',
                        'Property',
                        'Age in years',
                        'Other installment plans',
                        'Housing',
                        'Number of existing credits at this bank',
                        'Job',
                        'Number of people being liable to provide maintenance for',
                        'Telephone',
                        'foreign worker',
                        'label'
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                            'Status of existing checking account', #nominal
                            'Credit history', #nominal
                            'Purpose', #nominal
                            'Savings account/bonds', #nominal
                            'Present employment since', #nominal
                            'Personal status and sex', #nominal
                            'Other debtors / guarantors', #nominal
                            'Property', #nominal
                            'Other installment plans', #nominal
                            'Housing', #nominal
                            'Job', #nominal
                            'Telephone', #binary
                            'foreign worker', #binary
                            ]

        X_data = data.drop(['label'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['label'].values.reshape(-1, 1)).flatten(), name='label')            
 
    elif identifier == 'BIN:Banknote Authentication':

        feature_names = [
                        'variance',# of Wavelet Transformed image (continuous)
                        'skewness',# of Wavelet Transformed image (continuous)
                        'curtosis',# of Wavelet Transformed image (continuous)
                        'entropy',# of image (continuous)
                        'class',# (integer)
                        ]

        data = pd.read_csv('./real_world_datasets/Banknote/data_banknote_authentication.txt', names=feature_names, index_col=False)#, delimiter=' ')#, header=0)

        features_select = [
                        'variance',# of Wavelet Transformed image (continuous)
                        'skewness',# of Wavelet Transformed image (continuous)
                        'curtosis',# of Wavelet Transformed image (continuous)
                        'entropy',# of image (continuous)
                        'class',# (integer)
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='class')

    elif identifier == 'MULT:Contraceptive':

        feature_names = [
                        "Wife's age",# (numerical)
                        "Wife's education",# (categorical) 1=low, 2, 3, 4=high
                        "Husband's education",# (categorical) 1=low, 2, 3, 4=high
                        "Number of children ever born",# (numerical)
                        "Wife's religion",# (binary) 0=Non-Islam, 1=Islam
                        "Wife's now working?",# (binary) 0=Yes, 1=No
                        "Husband's occupation",# (categorical) 1, 2, 3, 4
                        "Standard-of-living index",# (categorical) 1=low, 2, 3, 4=high
                        "Media exposure",# (binary) 0=Good, 1=Not good
                        "Contraceptive method used",# (class attribute) 1=No-use, 2=Long-term, 3=Short-term
                        ]

        data = pd.read_csv('./real_world_datasets/Contraceptive/cmc.data', names=feature_names, index_col=False)#, delimiter=' ')#, header=0)

        features_select = [
                        "Wife's age",# (numerical)
                        "Wife's education",# (categorical) 1=low, 2, 3, 4=high
                        "Husband's education",# (categorical) 1=low, 2, 3, 4=high
                        "Number of children ever born",# (numerical)
                        "Wife's religion",# (binary) 0=Non-Islam, 1=Islam
                        "Wife's now working?",# (binary) 0=Yes, 1=No
                        "Husband's occupation",# (categorical) 1, 2, 3, 4
                        "Standard-of-living index",# (categorical) 1=low, 2, 3, 4=high
                        "Media exposure",# (binary) 0=Good, 1=Not good
                        "Contraceptive method used",# (class attribute) 1=No-use, 2=Long-term, 3=Short-term
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['Contraceptive method used'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Contraceptive method used'].values.reshape(-1, 1)).flatten(), name='Contraceptive method used')
        
    elif identifier == 'MULT:Segment':

        feature_names = ['target',
                        'region-centroid-col',#: the column of the center pixel of the region.
                        'region-centroid-row',#: the row of the center pixel of the region.
                        'region-pixel-count',#: the number of pixels in a region = 9.
                        'short-line-density-5',#: the results of a line extractoin algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region.
                        'short-line-density-2',#: same as short-line-density-5 but counts lines of high contrast, greater than 5.
                        'vedge-mean',#: measure the contrast of horizontally adjacent pixels in the region. There are 6, the mean and standard deviation are given. This attribute is used as a vertical edge detector.
                        'vegde-sd',#: (see 6)
                        'hedge-mean',#: measures the contrast of vertically adjacent pixels. Used for horizontal line detection.
                        'hedge-sd',#: (see 8).
                        'intensity-mean',#: the average over the region of (R + G + B)/3
                        'rawred-mean',#: the average over the region of the R value.
                        'rawblue-mean',#: the average over the region of the B value.
                        'rawgreen-mean',#: the average over the region of the G value.
                        'exred-mean',#: measure the excess red: (2R - (G + B))
                        'exblue-mean',#: measure the excess blue: (2B - (G + R))
                        'exgreen-mean',#: measure the excess green: (2G - (R + B))
                        'value-mean',#: 3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics)
                        'saturatoin-mean',#: (see 17)
                        'hue-mean',#: (see 17)
                        ]

        data1 = pd.read_csv('./real_world_datasets/Segment/segmentation.data', names=feature_names, index_col=False, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data2 = pd.read_csv('./real_world_datasets/Segment/segmentation.test', names=feature_names, index_col=False, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data = pd.concat([data1, data2])
        
        features_select = ['target',
                        'region-centroid-col',#: the column of the center pixel of the region.
                        'region-centroid-row',#: the row of the center pixel of the region.
                        'region-pixel-count',#: the number of pixels in a region = 9.
                        'short-line-density-5',#: the results of a line extractoin algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region.
                        'short-line-density-2',#: same as short-line-density-5 but counts lines of high contrast, greater than 5.
                        'vedge-mean',#: measure the contrast of horizontally adjacent pixels in the region. There are 6, the mean and standard deviation are given. This attribute is used as a vertical edge detector.
                        'vegde-sd',#: (see 6)
                        'hedge-mean',#: measures the contrast of vertically adjacent pixels. Used for horizontal line detection.
                        'hedge-sd',#: (see 8).
                        'intensity-mean',#: the average over the region of (R + G + B)/3
                        'rawred-mean',#: the average over the region of the R value.
                        'rawblue-mean',#: the average over the region of the B value.
                        'rawgreen-mean',#: the average over the region of the G value.
                        'exred-mean',#: measure the excess red: (2R - (G + B))
                        'exblue-mean',#: measure the excess blue: (2B - (G + R))
                        'exgreen-mean',#: measure the excess green: (2G - (R + B))
                        'value-mean',#: 3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics)
                        'saturatoin-mean',#: (see 17)
                        'hue-mean',#: (see 17)
                        ]

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['target'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['target'].values.reshape(-1, 1)).flatten(), name='target')        
              
    elif identifier == 'BIN:Spambase':

        feature_names = flatten_list([
                                    ['word_freq_WORD_' + str(i) for i in range(48)],
                                    ['char_freq_CHAR_' + str(i) for i in range(6)],
                                    'capital_run_length_average',
                                    'capital_run_length_longest',
                                    'capital_run_length_total',
                                    'spam_type'
                                    ])

        data = pd.read_csv('./real_world_datasets/Spambase/spambase.data', names=feature_names, index_col=False)#, header=2)#, header=0)#, delimiter=' ')#, header=0)

        
        features_select = flatten_list([
                                    ['word_freq_WORD_' + str(i) for i in range(48)],
                                    ['char_freq_CHAR_' + str(i) for i in range(6)],
                                    'capital_run_length_average',
                                    'capital_run_length_longest',
                                    'capital_run_length_total',
                                    'spam_type'
                                    ])

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['spam_type'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['spam_type'].values.reshape(-1, 1)).flatten(), name='spam_type')
                
    elif identifier == 'MULT:Optical Recognition':

        feature_names = flatten_list([
                                    [str(i) for i in range(64)],
                                    'class_code'
                                    ])

        data1 = pd.read_csv('./real_world_datasets/Optical Recognition/optdigits.tra', names=feature_names, index_col=False)#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data2 = pd.read_csv('./real_world_datasets/Optical Recognition/optdigits.tes', names=feature_names, index_col=False)#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data = pd.concat([data1, data2])
        
        features_select = flatten_list([
                                    [str(i) for i in range(64)],
                                    'class_code'
                                    ])

        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['class_code'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class_code'].values.reshape(-1, 1)).flatten(), name='class_code')        

    elif identifier == 'MULT:Landsat':

        feature_names = flatten_list([
                                    [['spectral_' + str(i) + '_pixel_' + str(j) for i in range(4)] for j in range(9)],   
                                    'target',
                        ])

        data1 = pd.read_csv('./real_world_datasets/Landsat/sat.trn', names=feature_names, index_col=False, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data2 = pd.read_csv('./real_world_datasets/Landsat/sat.tst', names=feature_names, index_col=False, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data = pd.concat([data1, data2])
        
        features_select = flatten_list([
                                    [['spectral_' + str(i) + '_pixel_' + str(j) for i in range(4)] for j in range(9)],   
                                    'target',
                        ])


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['target'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['target'].values.reshape(-1, 1)).flatten(), name='target')        
        
    elif identifier == 'MULT:Pendigits':

        feature_names = flatten_list([
                                    [str(i) for i in range(16)],   
                                    'number',
                        ])

        data1 = pd.read_csv('./real_world_datasets/Pendigits/pendigits.tra', names=feature_names, index_col=False)#, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data2 = pd.read_csv('./real_world_datasets/Pendigits/pendigits.tes', names=feature_names, index_col=False)#, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        data = pd.concat([data1, data2])
        
        features_select = flatten_list([
                                    [str(i) for i in range(16)],   
                                    'number',
                        ])


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['number'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['number'].values.reshape(-1, 1)).flatten(), name='number')               
        
    elif identifier == 'MULT:Letter':

        feature_names = [
                        'lettr',# capital letter (26 values from A to Z)
                        'x-box',# horizontal position of box (integer)
                        'y-box',# vertical position of box (integer)
                        'width',# width of box (integer)
                        'high',# height of box (integer)
                        'onpix',# total # on pixels (integer)
                        'x-bar',# mean x of on pixels in box (integer)
                        'y-bar',# mean y of on pixels in box (integer)
                        'x2bar',# mean x variance (integer)
                        'y2bar',# mean y variance (integer)
                        'xybar',# mean x y correlation (integer)
                        'x2ybr',# mean of x * x * y (integer)
                        'xy2br',# mean of x * y * y (integer)
                        'x-ege',# mean edge count left to right (integer)
                        'xegvy',# correlation of x-ege with y (integer)
                        'y-ege',# mean edge count bottom to top (integer)
                        'yegvx',# correlation of y-ege with x (integer)
                        ]

        data = pd.read_csv('./real_world_datasets/Letter/letter-recognition.data', names=feature_names, index_col=False)#, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        
        features_select = [
                        'lettr',# capital letter (26 values from A to Z)
                        'x-box',# horizontal position of box (integer)
                        'y-box',# vertical position of box (integer)
                        'width',# width of box (integer)
                        'high',# height of box (integer)
                        'onpix',# total # on pixels (integer)
                        'x-bar',# mean x of on pixels in box (integer)
                        'y-bar',# mean y of on pixels in box (integer)
                        'x2bar',# mean x variance (integer)
                        'y2bar',# mean y variance (integer)
                        'xybar',# mean x y correlation (integer)
                        'x2ybr',# mean of x * x * y (integer)
                        'xy2br',# mean of x * y * y (integer)
                        'x-ege',# mean edge count left to right (integer)
                        'xegvy',# correlation of x-ege with y (integer)
                        'y-ege',# mean edge count bottom to top (integer)
                        'yegvx',# correlation of y-ege with x (integer)
                        ]


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['lettr'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['lettr'].values.reshape(-1, 1)).flatten(), name='lettr')               

    elif identifier == 'MULT:Connect4':

        feature_names = [
                        'a1',#: {x,o,b}
                        'a2',#: {x,o,b}
                        'a3',#: {x,o,b}
                        'a4',#: {x,o,b}
                        'a5',#: {x,o,b}
                        'a6',#: {x,o,b}
                        'b1',#: {x,o,b}
                        'b2',#: {x,o,b}
                        'b3',#: {x,o,b}
                        'b4',#: {x,o,b}
                        'b5',#: {x,o,b}
                        'b6',#: {x,o,b}
                        'c1',#: {x,o,b}
                        'c2',#: {x,o,b}
                        'c3',#: {x,o,b}
                        'c4',#: {x,o,b}
                        'c5',#: {x,o,b}
                        'c6',#: {x,o,b}
                        'd1',#: {x,o,b}
                        'd2',#: {x,o,b}
                        'd3',#: {x,o,b}
                        'd4',#: {x,o,b}
                        'd5',#: {x,o,b}
                        'd6',#: {x,o,b}
                        'e1',#: {x,o,b}
                        'e2',#: {x,o,b}
                        'e3',#: {x,o,b}
                        'e4',#: {x,o,b}
                        'e5',#: {x,o,b}
                        'e6',#: {x,o,b}
                        'f1',#: {x,o,b}
                        'f2',#: {x,o,b}
                        'f3',#: {x,o,b}
                        'f4',#: {x,o,b}
                        'f5',#: {x,o,b}
                        'f6',#: {x,o,b}
                        'g1',#: {x,o,b}
                        'g2',#: {x,o,b}
                        'g3',#: {x,o,b}
                        'g4',#: {x,o,b}
                        'g5',#: {x,o,b}
                        'g6',#: {x,o,b}
                        'Class',#: {win,loss,draw}
                        ]

        data = pd.read_csv('./real_world_datasets/Connect4/connect-4.data', names=feature_names, delimiter=',')#, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)

        features_select = [
                        'a1',#: {x,o,b}
                        'a2',#: {x,o,b}
                        'a3',#: {x,o,b}
                        'a4',#: {x,o,b}
                        'a5',#: {x,o,b}
                        'a6',#: {x,o,b}
                        'b1',#: {x,o,b}
                        'b2',#: {x,o,b}
                        'b3',#: {x,o,b}
                        'b4',#: {x,o,b}
                        'b5',#: {x,o,b}
                        'b6',#: {x,o,b}
                        'c1',#: {x,o,b}
                        'c2',#: {x,o,b}
                        'c3',#: {x,o,b}
                        'c4',#: {x,o,b}
                        'c5',#: {x,o,b}
                        'c6',#: {x,o,b}
                        'd1',#: {x,o,b}
                        'd2',#: {x,o,b}
                        'd3',#: {x,o,b}
                        'd4',#: {x,o,b}
                        'd5',#: {x,o,b}
                        'd6',#: {x,o,b}
                        'e1',#: {x,o,b}
                        'e2',#: {x,o,b}
                        'e3',#: {x,o,b}
                        'e4',#: {x,o,b}
                        'e5',#: {x,o,b}
                        'e6',#: {x,o,b}
                        'f1',#: {x,o,b}
                        'f2',#: {x,o,b}
                        'f3',#: {x,o,b}
                        'f4',#: {x,o,b}
                        'f5',#: {x,o,b}
                        'f6',#: {x,o,b}
                        'g1',#: {x,o,b}
                        'g2',#: {x,o,b}
                        'g3',#: {x,o,b}
                        'g4',#: {x,o,b}
                        'g5',#: {x,o,b}
                        'g6',#: {x,o,b}
                        'Class',#: {win,loss,draw}
                        ]


        data = data[features_select]

        nominal_features = [
                            ]
        ordinal_features = [    
                            'a1',#: {x,o,b}
                            'a2',#: {x,o,b}
                            'a3',#: {x,o,b}
                            'a4',#: {x,o,b}
                            'a5',#: {x,o,b}
                            'a6',#: {x,o,b}
                            'b1',#: {x,o,b}
                            'b2',#: {x,o,b}
                            'b3',#: {x,o,b}
                            'b4',#: {x,o,b}
                            'b5',#: {x,o,b}
                            'b6',#: {x,o,b}
                            'c1',#: {x,o,b}
                            'c2',#: {x,o,b}
                            'c3',#: {x,o,b}
                            'c4',#: {x,o,b}
                            'c5',#: {x,o,b}
                            'c6',#: {x,o,b}
                            'd1',#: {x,o,b}
                            'd2',#: {x,o,b}
                            'd3',#: {x,o,b}
                            'd4',#: {x,o,b}
                            'd5',#: {x,o,b}
                            'd6',#: {x,o,b}
                            'e1',#: {x,o,b}
                            'e2',#: {x,o,b}
                            'e3',#: {x,o,b}
                            'e4',#: {x,o,b}
                            'e5',#: {x,o,b}
                            'e6',#: {x,o,b}
                            'f1',#: {x,o,b}
                            'f2',#: {x,o,b}
                            'f3',#: {x,o,b}
                            'f4',#: {x,o,b}
                            'f5',#: {x,o,b}
                            'f6',#: {x,o,b}
                            'g1',#: {x,o,b}
                            'g2',#: {x,o,b}
                            'g3',#: {x,o,b}
                            'g4',#: {x,o,b}
                            'g5',#: {x,o,b}
                            'g6',#: {x,o,b}            
                            ]

        X_data = data.drop(['Class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class'].values.reshape(-1, 1)).flatten(), name='Class')     
   
    elif identifier == 'MULT:MNIST-pixels':

        X_data1, y_data1 = _read_idx_file('./real_world_datasets/MNIST_pixels/mnist', 780, ' ')#pd.concat([data1, data2])
        X_data2, y_data2 = _read_idx_file('./real_world_datasets/MNIST_pixels/mnist.t', 780, ' ')#pd.concat([data1, data2])
        
        
        nominal_features = [
                            ]
        ordinal_features = [              
                            ]        
        
        X_data = np.concatenate([X_data1, X_data2])
        y_data = np.concatenate([y_data1, y_data2])
        
        X_data = pd.DataFrame(data=X_data)
        y_data = pd.Series(y_data, name='label')  
        
    elif identifier == 'MULT:SensIT':

        X_data1, y_data1 = _read_idx_file('./real_world_datasets/SensIT/combined', 100, ' ')#pd.concat([data1, data2])
        X_data2, y_data2 = _read_idx_file('./real_world_datasets/SensIT/combined.t', 100, ' ')#pd.concat([data1, data2])
        
        nominal_features = [
                            ]
        ordinal_features = [              
                            ]   
        
        X_data = np.concatenate([X_data1, X_data2])
        y_data = np.concatenate([y_data1, y_data2])
        
        X_data = pd.DataFrame(data=X_data)
        y_data = pd.Series(y_data, name='label')          
        
    elif identifier == 'MULT:MNIST-LeNet5':

        
        mat = scipy.io.loadmat('./real_world_datasets/MNIST_LeNet5/MNIST_LeNet5.mat')
        
        X_data1, y_data1 = mat['train_fea'], mat['train_gnd'].reshape(-1,) 
        X_data2, y_data2 = mat['test_fea'], mat['test_gnd'].reshape(-1,) 
        
        nominal_features = [
                            ]
        ordinal_features = [              
                            ]           
        
        X_data = np.concatenate([X_data1, X_data2])
        y_data = np.concatenate([y_data1, y_data2])
        
        X_data = pd.DataFrame(data=X_data)
        y_data = pd.Series(y_data, name='label')          
        
    elif identifier == 'MULT:MNIST':
        
        feature_names = [i for i in range(65)]

        data1 = pd.read_csv('./real_world_datasets/Optical Recognition/optdigits.tra', names=feature_names, index_col=False)#, delimiter=' ', header=0)
        data2 = pd.read_csv('./real_world_datasets/Optical Recognition/optdigits.tes', names=feature_names, index_col=False)#, delimiter=' ', header=0)

        data = pd.concat([data1, data2])

        features_select = [i for i in range(65)]

        data = data[features_select]

        nominal_features = [  
                            ]
        ordinal_features = [  
                            ]

        X_data = data.drop([64], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data[64].values.reshape(-1, 1)).flatten(), name='number')   

    elif identifier == 'MULT:protein':

        X_data, y_data = _read_idx_file('./real_world_datasets/Protein/protein', 357, '  ')#pd.concat([data1, data2])

        nominal_features = [
                            ]
        ordinal_features = [              
                            ]           
        
        X_data = pd.DataFrame(data=X_data)
        y_data = pd.Series(y_data, name='label')   

    elif identifier == 'REG:boston':
            
        feature_names = [
                            'CRIM',#: per capita crime rate by town
                            'ZN',#: proportion of residential land zoned for lots over 25,000 sq.ft.
                            'INDUS',#: proportion of non-retail business acres per town
                            'CHAS',#: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                            'NOX',#: nitric oxides concentration (parts per 10 million)
                            'RM',#: average number of rooms per dwelling
                            'AGE',#: proportion of owner-occupied units built prior to 1940
                            'DIS',#: weighted distances to five Boston employment centres
                            'RAD',#: index of accessibility to radial highways
                            'TAX',#: full-value property-tax rate per $10,000
                            'PTRATIO',#: pupil-teacher ratio by town
                            'B',#: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
                            'LSTAT',#: % lower status of the population
                            'MEDV',#: Median value of owner-occupied homes in $1000's            
                        ]

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', names=feature_names, index_col=False, delimiter='\s+')#, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        
        features_select = [
                            'CRIM',#: per capita crime rate by town
                            'ZN',#: proportion of residential land zoned for lots over 25,000 sq.ft.
                            'INDUS',#: proportion of non-retail business acres per town
                            'CHAS',#: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                            'NOX',#: nitric oxides concentration (parts per 10 million)
                            'RM',#: average number of rooms per dwelling
                            'AGE',#: proportion of owner-occupied units built prior to 1940
                            'DIS',#: weighted distances to five Boston employment centres
                            'RAD',#: index of accessibility to radial highways
                            'TAX',#: full-value property-tax rate per $10,000
                            'PTRATIO',#: pupil-teacher ratio by town
                            'B',#: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
                            'LSTAT',#: % lower status of the population
                            'MEDV',#: Median value of owner-occupied homes in $1000's                

                        ]


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['MEDV'], axis = 1)
        y_data = data['MEDV']
            
            
    elif identifier == 'REG:Medical Insurance':
        
        data = pd.read_csv('real_world_datasets/Medical Insurance/insurance.csv', delimiter=',')

        features_select = [
                            'age', 
                            'sex', 
                            'bmi', 
                            'children', 
                            'smoker',
                            'region',
                            'charges'
                            ]

        data = data[features_select]

        nominal_features = [
                            ]
        ordinal_features = [
                            'region',
                            'sex',
                            'smoker'
                           ]


        X_data = data.drop(['charges'], axis = 1)
        y_data = data['charges']
        

    elif identifier == 'REG:concrete':
            
        data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')#, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)


        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['Concrete compressive strength(MPa, megapascals) '], axis = 1)
        y_data = data['Concrete compressive strength(MPa, megapascals) ']
        
                    
    elif identifier == 'REG:airfoil':
            
        feature_names = [
                        'Frequency',#, in Hertzs.
                        'Angle of attack',#, in degrees.
                        'Chord length',#, in meters.
                        'Free-stream velocity',#, in meters per second.
                        'Suction side displacement thickness',#, in meters.
                        'Scaled sound pressure level',#, in decibels.  
                        ]

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat', names=feature_names, index_col=False, delimiter='\t')#, delimiter=' ')#, header=2)#, header=0)#, delimiter=' ')#, header=0)
        
        features_select = [
                        'Frequency',#, in Hertzs.
                        'Angle of attack',#, in degrees.
                        'Chord length',#, in meters.
                        'Free-stream velocity',#, in meters per second.
                        'Suction side displacement thickness',#, in meters.
                        'Scaled sound pressure level',#, in decibels.    
                        ]


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['Scaled sound pressure level'], axis = 1)
        y_data = data['Scaled sound pressure level']  

    elif identifier == 'REG:Abalone':
        
        feature_names = [
                        'Sex',#		nominal			M, F, and I (infant)
                        'Length',#	continuous	mm	Longest shell measurement
                        'Diameter',#	continuous	mm	perpendicular to length
                        'Height',#		continuous	mm	with meat in shell
                        'Whole weight',#	continuous	grams	whole abalone
                        'Shucked weight',#	continuous	grams	weight of meat
                        'Viscera weight',#	continuous	grams	gut weight (after bleeding)
                        'Shell weight',#	continuous	grams	after being dried
                        'Rings',#		integer			+1.5 gives the age in years
                        ]

        data = pd.read_csv('./real_world_datasets/Abalone/abalone.data', names=feature_names, index_col=False)


        features_select = [
                        'Sex',#		nominal			M, F, and I (infant)
                        'Length',#	continuous	mm	Longest shell measurement
                        'Diameter',#	continuous	mm	perpendicular to length
                        'Height',#		continuous	mm	with meat in shell
                        'Whole weight',#	continuous	grams	whole abalone
                        'Shucked weight',#	continuous	grams	weight of meat
                        'Viscera weight',#	continuous	grams	gut weight (after bleeding)
                        'Shell weight',#	continuous	grams	after being dried
                        'Rings',#		integer			+1.5 gives the age in years
                            ]

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                                'Sex',
                           ]

        X_data = data.drop(['Rings'], axis = 1)
        y_data = data['Rings'].astype(np.float32)#((data['Rings'] > 10) * 1)         
        
        
    elif identifier == 'REG:cpuact':
        # https://www.openml.org/search?type=data&sort=runs&id=197&status=active

        raw_data = loadarff('./real_world_datasets/cpuact/dataset_2183_cpu_act.arff')
        data = pd.DataFrame(raw_data[0])

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['usr'], axis = 1)
        y_data = data['usr']          
        
        
    elif identifier == 'REG:ailerons':
        # https://www.openml.org/search?type=data&sort=runs&id=296&status=active

        raw_data = loadarff('./real_world_datasets/ailerons/ailerons.arff')
        data = pd.DataFrame(raw_data[0])

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['goal'], axis = 1)
        y_data = data['goal']        
        
        
    elif identifier == 'REG:CT slice':

        feature_names = flatten_list([
                        'patientId',
                        ['bone_hist_' + str(i) for i in range(240)],
                        ['air_inclusion_hist_' + str(i) for i in range(144)],
                        'reference', # Relative location of the image on the axial axis (class value). Values are in the range [0; 180] where 0 denotes the top of the head and 180 the soles of the feet.
                        ])
    
        data = pd.read_csv('./real_world_datasets/ct_slice/slice_localization_data.csv', names=feature_names, index_col=False, header=0)



        features_select = flatten_list([
                        'patientId',
                        ['bone_hist_' + str(i) for i in range(240)],
                        ['air_inclusion_hist_' + str(i) for i in range(144)],
                        'reference', # Relative location of the image on the axial axis (class value). Values are in the range [0; 180] where 0 denotes the top of the head and 180 the soles of the feet.
                        ])

        data = data[features_select]

        nominal_features = [
                                ]
        ordinal_features = [
                           ]

        X_data = data.drop(['reference'], axis = 1)
        y_data = data['reference']
        
    elif identifier == 'REG:YearPredictionMSD':
            
        feature_names = flatten_list([
                        'year', #ranging from 1922 to 2011.
                        ['timbre_average_' + str(i) for i in range(12)],
                        ['timbre_covariance_' + str(i) for i in range(78)],
                        ])

        data = pd.read_csv('./real_world_datasets/YearPredictionMSD/YearPredictionMSD.txt', names=feature_names, index_col=False)
        
        features_select = flatten_list([
                        'year', #ranging from 1922 to 2011.
                        ['timbre_average_' + str(i) for i in range(12)],
                        ['timbre_covariance_' + str(i) for i in range(78)],
                        ])


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['year'], axis = 1)
        y_data = data['year'].astype(np.float32)    
        
    
    elif identifier == 'BIN:Mushroom':
        feature_names = [
                        'eadible',#
                        'cap-shape',#: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
                        'cap-surface',#: fibrous=f,grooves=g,scaly=y,smooth=s
                        'cap-color',#: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
                        'ruises?',#: bruises=t,no=f
                        'odor',#: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
                        'gill-attachment',#: attached=a,descending=d,free=f,notched=n
                        'gill-spacing',#: close=c,crowded=w,distant=d
                        'gill-size',#: broad=b,narrow=n
                        'gill-color',#: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
                        'stalk-shape',#: enlarging=e,tapering=t
                        'stalk-root',#: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
                        'stalk-surface-above-ring',#: fibrous=f,scaly=y,silky=k,smooth=s
                        'stalk-surface-below-ring',#: fibrous=f,scaly=y,silky=k,smooth=s
                        'stalk-color-above-ring',#: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
                        'stalk-color-below-ring',#: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
                        'veil-type',#: partial=p,universal=u
                        'veil-color',#: brown=n,orange=o,white=w,yellow=y
                        'ring-number',#: none=n,one=o,two=t
                        'ring-type',#: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
                        'spore-print-color',#: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
                        'population',#: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
                        'habitat',#: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
                        ]      
        
        data = pd.read_csv('./real_world_datasets/Mushroom/agaricus-lepiota.data', names=feature_names, index_col=False)
        
        features_select = [
                        'eadible',#
                        'cap-shape',#: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
                        'cap-surface',#: fibrous=f,grooves=g,scaly=y,smooth=s
                        'cap-color',#: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
                        'ruises?',#: bruises=t,no=f
                        'odor',#: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
                        'gill-attachment',#: attached=a,descending=d,free=f,notched=n
                        'gill-spacing',#: close=c,crowded=w,distant=d
                        'gill-size',#: broad=b,narrow=n
                        'gill-color',#: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
                        'stalk-shape',#: enlarging=e,tapering=t
                        'stalk-root',#: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
                        'stalk-surface-above-ring',#: fibrous=f,scaly=y,silky=k,smooth=s
                        'stalk-surface-below-ring',#: fibrous=f,scaly=y,silky=k,smooth=s
                        'stalk-color-above-ring',#: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
                        'stalk-color-below-ring',#: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
                        'veil-type',#: partial=p,universal=u
                        'veil-color',#: brown=n,orange=o,white=w,yellow=y
                        'ring-number',#: none=n,one=o,two=t
                        'ring-type',#: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
                        'spore-print-color',#: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
                        'population',#: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
                        'habitat',#: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                        'cap-shape',#: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
                        'cap-surface',#: fibrous=f,grooves=g,scaly=y,smooth=s
                        'cap-color',#: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
                        'ruises?',#: bruises=t,no=f
                        'odor',#: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
                        'gill-attachment',#: attached=a,descending=d,free=f,notched=n
                        'gill-spacing',#: close=c,crowded=w,distant=d
                        'gill-size',#: broad=b,narrow=n
                        'gill-color',#: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
                        'stalk-shape',#: enlarging=e,tapering=t
                        'stalk-root',#: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
                        'stalk-surface-above-ring',#: fibrous=f,scaly=y,silky=k,smooth=s
                        'stalk-surface-below-ring',#: fibrous=f,scaly=y,silky=k,smooth=s
                        'stalk-color-above-ring',#: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
                        'stalk-color-below-ring',#: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
                        'veil-type',#: partial=p,universal=u
                        'veil-color',#: brown=n,orange=o,white=w,yellow=y
                        'ring-number',#: none=n,one=o,two=t
                        'ring-type',#: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
                        'spore-print-color',#: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
                        'population',#: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
                        'habitat',#: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d            
                            ]


        X_data = data.drop(['eadible'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['eadible'].values.reshape(-1, 1)).flatten(), name='number')            
                            
    elif identifier == 'BIN:Raisins':  
        
        data = pd.read_excel('./real_world_datasets/Raisins/Raisin_Dataset.xlsx')

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['Class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class'].values.reshape(-1, 1)).flatten(), name='number')          
            
    elif identifier == 'BIN:Rice':
     
        
        data = pd.read_excel('./real_world_datasets/Rice/Rice_Cammeo_Osmancik.xlsx')

        nominal_features = [
                        ]
        ordinal_features = [          
                            ]

        X_data = data.drop(['Class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class'].values.reshape(-1, 1)).flatten(), name='number')       
        
    elif identifier == 'BIN:Horse Colic':
        feature_names = [
                        'surgery',
                        'Age',
                        'Hospital Number',
                        'rectal temperature',
                        'pulse',
                        'respiratory rate',
                        'temperature of extremities',
                        'peripheral pulse',
                        'mucous membranes',
                        'capillary refill time',
                        'pain',
                        'peristalsis',
                        'abdominal distension',
                        'nasogastric tube',
                        'nasogastric reflux',
                        'nasogastric reflux PH',
                        'rectal examination',
                        'abdomen',
                        'packed cell volume',
                        'total protein',
                        'abdominocentesis appearance',
                        'abdomcentesis total protein',
                        'outcome',
                        'surgical lesion?', #TARGET
                        'type of lesion1',
                        'type of lesion2',
                        'type of lesion3',
                        'cp_data',         
            
                        ]      
        
        data1 = pd.read_csv('./real_world_datasets/Colic/horse-colic.data', names=feature_names, index_col=False, delimiter=' ', na_values='?')
        data2 = pd.read_csv('./real_world_datasets/Colic/horse-colic.test', names=feature_names, index_col=False, delimiter=' ', na_values='?')
        
        data = pd.concat([data1, data2])   
        data = data.fillna(data.mode().iloc[0])
        features_select = [
                        'surgery',
                        'Age',
                        'Hospital Number',
                        'rectal temperature',
                        'pulse',
                        'respiratory rate',
                        'temperature of extremities',
                        'peripheral pulse',
                        'mucous membranes',
                        'capillary refill time',
                        'pain',
                        'peristalsis',
                        'abdominal distension',
                        'nasogastric tube',
                        'nasogastric reflux',
                        'nasogastric reflux PH',
                        'rectal examination',
                        'abdomen',
                        'packed cell volume',
                        'total protein',
                        'abdominocentesis appearance',
                        'abdomcentesis total protein',
                        'outcome',
                        'surgical lesion?', #TARGET
                        'type of lesion1',
                        'type of lesion2',
                        'type of lesion3',
                        'cp_data',
                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
           
                            ]

        X_data = data.drop(['surgical lesion?'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['surgical lesion?'].values.reshape(-1, 1)).flatten(), name='number')                 
                    
    elif identifier == 'BIN:Echocardiogram':
        
        feature_names = [
                        'survival',# -- the number of months patient survived (has survived, if patient is still alive). Because all the patients had their heart attacks at different times, it is possible that some patients have survived less than one year but they are still alive. Check the second variable to confirm this. Such patients cannot be used for the prediction task mentioned above.
                        'still-alive',# -- a binary variable. 0=dead at end of survival period, 1 means still alive
                        'age-at-heart-attack',# -- age in years when heart attack occurred
                        'pericardial-effusion',# -- binary. Pericardial effusion is fluid around the heart. 0=no fluid, 1=fluid
                        'fractional-shortening',# -- a measure of contracility around the heart lower numbers are increasingly abnormal
                        'epss',# -- E-point septal separation, another measure of contractility. Larger numbers are increasingly abnormal.
                        'lvdd',# -- left ventricular end-diastolic dimension. This is a measure of the size of the heart at end-diastole. Large hearts tend to be sick hearts.
                        'wall-motion-score',# -- a measure of how the segments of the left ventricle are moving
                        'wall-motion-index',# -- equals wall-motion-score divided by number of segments seen. Usually 12-13 segments are seen in an echocardiogram. Use this variable INSTEAD of the wall motion score.
                        'mult',# -- a derivate var which can be ignored
                        'name',# -- the name of the patient (I have replaced them with "name")
                        'group',# -- meaningless, ignore it
                        'alive-at-1',# -- Boolean-valued. Derived from the first two attributes. 0 means patient was either dead after 1 year or had been followed for less than 1 year. 1 means patient was alive at 1 year.            
                        ]      
        
        data = pd.read_csv('./real_world_datasets/Echocardiogram/echocardiogram.data', names=feature_names, index_col=False, na_values='?')
        data = data.fillna(data.mode().iloc[0])
        
        features_select = [
                        #'survival',# -- the number of months patient survived (has survived, if patient is still alive). Because all the patients had their heart attacks at different times, it is possible that some patients have survived less than one year but they are still alive. Check the second variable to confirm this. Such patients cannot be used for the prediction task mentioned above.
                        #'still-alive',# -- a binary variable. 0=dead at end of survival period, 1 means still alive
                        'age-at-heart-attack',# -- age in years when heart attack occurred
                        'pericardial-effusion',# -- binary. Pericardial effusion is fluid around the heart. 0=no fluid, 1=fluid
                        'fractional-shortening',# -- a measure of contracility around the heart lower numbers are increasingly abnormal
                        'epss',# -- E-point septal separation, another measure of contractility. Larger numbers are increasingly abnormal.
                        'lvdd',# -- left ventricular end-diastolic dimension. This is a measure of the size of the heart at end-diastole. Large hearts tend to be sick hearts.
                        'wall-motion-score',# -- a measure of how the segments of the left ventricle are moving
                        'wall-motion-index',# -- equals wall-motion-score divided by number of segments seen. Usually 12-13 segments are seen in an echocardiogram. Use this variable INSTEAD of the wall motion score.
                        'mult',# -- a derivate var which can be ignored
                        #'name',# -- the name of the patient (I have replaced them with "name")
                        #'group',# -- meaningless, ignore it
                        'alive-at-1',# -- Boolean-valued. Derived from the first two attributes. 0 means patient was either dead after 1 year or had been followed for less than 1 year. 1 means patient was alive at 1 year.            
            

                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
           
                            ]


        X_data = data.drop(['alive-at-1'], axis = 1)
        y_data = ((data['alive-at-1'] != 0) * 1)#pd.Series(OrdinalEncoder().fit_transform(data['alive-at-1'].values.reshape(-1, 1)).flatten(), name='number')                
       
    elif identifier == 'BIN:Thyroid':
        feature_names = [
                    'age',#:				continuous.
                    'sex',#:				M, F.
                    'on thyroxine',#:			f, t.
                    'query on thyroxine',#:		f, t.
                    'on antithyroid medication',#:	f, t.
                    'sick',#:				f, t.
                    'pregnant',#:			f, t.
                    'thyroid surgery',#:		f, t.
                    'I131 treatment',#:			f, t.
                    'query hypothyroid',#:		f, t.
                    'query hyperthyroid',#:		f, t.
                    'lithium',#:			f, t.
                    'goitre',#:				f, t.
                    'tumor',#:				f, t.
                    'hypopituitary',#:			f, t.
                    'psych',#:				f, t.
                    'TSH measured',#:			f, t.
                    'TSH',#:				continuous.
                    'T3 measured',#:			f, t.
                    'T3',#:				continuous.
                    'TT4 measured',#:			f, t.
                    'TT4',#:				continuous.
                    'T4U measured',#:			f, t.
                    'T4U',#:				continuous.
                    'FTI measured',#:			f, t.
                    'FTI',#:				continuous.
                    'TBG measured',#:			f, t.
                    'TBG',#:				continuous.
                    'referral source',#:		WEST, STMW, SVHC, SVI, SVHD, other. 
                    'class',
                        ]      
        
        data = pd.read_csv('./real_world_datasets/Thyroid/thyroid0387.data', names=feature_names, index_col=False, na_values='?')
        data = data.fillna(data.mode().iloc[0])
        
        features_select = [
                    'age',#:				continuous.
                    'sex',#:				M, F.
                    'on thyroxine',#:			f, t.
                    'query on thyroxine',#:		f, t.
                    'on antithyroid medication',#:	f, t.
                    'sick',#:				f, t.
                    'pregnant',#:			f, t.
                    'thyroid surgery',#:		f, t.
                    'I131 treatment',#:			f, t.
                    'query hypothyroid',#:		f, t.
                    'query hyperthyroid',#:		f, t.
                    'lithium',#:			f, t.
                    'goitre',#:				f, t.
                    'tumor',#:				f, t.
                    'hypopituitary',#:			f, t.
                    'psych',#:				f, t.
                    'TSH measured',#:			f, t.
                    'TSH',#:				continuous.
                    'T3 measured',#:			f, t.
                    'T3',#:				continuous.
                    'TT4 measured',#:			f, t.
                    'TT4',#:				continuous.
                    'T4U measured',#:			f, t.
                    'T4U',#:				continuous.
                    'FTI measured',#:			f, t.
                    'FTI',#:				continuous.
                    'TBG measured',#:			f, t.
                    'TBG',#:				continuous.
                    'referral source',#:		WEST, STMW, SVHC, SVI, SVHD, other. 
                    'class',            
                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                            'sex',#:				M, F.
                            'on thyroxine',#:			f, t.
                            'query on thyroxine',#:		f, t.
                            'on antithyroid medication',#:	f, t.
                            'sick',#:				f, t.
                            'pregnant',#:			f, t.
                            'thyroid surgery',#:		f, t.
                            'I131 treatment',#:			f, t.
                            'query hypothyroid',#:		f, t.
                            'query hyperthyroid',#:		f, t.
                            'lithium',#:			f, t.
                            'goitre',#:				f, t.
                            'tumor',#:				f, t.
                            'hypopituitary',#:			f, t.
                            'psych',#:				f, t.
                            'TSH measured',#:			f, t.
                            'T3 measured',#:			f, t.
                            'TT4 measured',#:			f, t.
                            'T4U measured',#:			f, t.
                            'FTI measured',#:			f, t.
                            'TBG measured',#:			f, t.
                            'referral source',#:		WEST, STMW, SVHC, SVI, SVHD, other. 
                            ]


        X_data = data.drop(['class'], axis = 1)
        y_data =  ((data['class'].str.contains('-')) * 1)#((data['class'] != '-') * 1)        #pd.Series(OrdinalEncoder().fit_transform(data['eadible'].values.reshape(-1, 1)).flatten(), name='number')                 
        
    elif identifier == 'BIN:Congressional Voting':
        feature_names = [
                        'Class Name',#: 2 (democrat, republican)
                        'handicapped-infants',#: 2 (y,n)
                        'water-project-cost-sharing',#: 2 (y,n)
                        'adoption-of-the-budget-resolution',#: 2 (y,n)
                        'physician-fee-freeze',#: 2 (y,n)
                        'el-salvador-aid',#: 2 (y,n)
                        'religious-groups-in-schools',#: 2 (y,n)
                        'anti-satellite-test-ban',#: 2 (y,n)
                        'aid-to-nicaraguan-contras',#: 2 (y,n)
                        'mx-missile',#: 2 (y,n)
                        'immigration',#: 2 (y,n)
                        'synfuels-corporation-cutback',#: 2 (y,n)
                        'education-spending',#: 2 (y,n)
                        'superfund-right-to-sue',#: 2 (y,n)
                        'crime',#: 2 (y,n)
                        'duty-free-exports',#: 2 (y,n)
                        'export-administration-act-south-africa',#: 2 (y,n)                   
                        ]      
        
        data = pd.read_csv('./real_world_datasets/Congressional Voting/house-votes-84.data', names=feature_names, index_col=False)
        
        features_select = [
                        'Class Name',#: 2 (democrat, republican)
                        'handicapped-infants',#: 2 (y,n)
                        'water-project-cost-sharing',#: 2 (y,n)
                        'adoption-of-the-budget-resolution',#: 2 (y,n)
                        'physician-fee-freeze',#: 2 (y,n)
                        'el-salvador-aid',#: 2 (y,n)
                        'religious-groups-in-schools',#: 2 (y,n)
                        'anti-satellite-test-ban',#: 2 (y,n)
                        'aid-to-nicaraguan-contras',#: 2 (y,n)
                        'mx-missile',#: 2 (y,n)
                        'immigration',#: 2 (y,n)
                        'synfuels-corporation-cutback',#: 2 (y,n)
                        'education-spending',#: 2 (y,n)
                        'superfund-right-to-sue',#: 2 (y,n)
                        'crime',#: 2 (y,n)
                        'duty-free-exports',#: 2 (y,n)
                        'export-administration-act-south-africa',#: 2 (y,n)    
                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                        'handicapped-infants',#: 2 (y,n)
                        'water-project-cost-sharing',#: 2 (y,n)
                        'adoption-of-the-budget-resolution',#: 2 (y,n)
                        'physician-fee-freeze',#: 2 (y,n)
                        'el-salvador-aid',#: 2 (y,n)
                        'religious-groups-in-schools',#: 2 (y,n)
                        'anti-satellite-test-ban',#: 2 (y,n)
                        'aid-to-nicaraguan-contras',#: 2 (y,n)
                        'mx-missile',#: 2 (y,n)
                        'immigration',#: 2 (y,n)
                        'synfuels-corporation-cutback',#: 2 (y,n)
                        'education-spending',#: 2 (y,n)
                        'superfund-right-to-sue',#: 2 (y,n)
                        'crime',#: 2 (y,n)
                        'duty-free-exports',#: 2 (y,n)
                        'export-administration-act-south-africa',#: 2 (y,n)                
                            ]


        X_data = data.drop(['Class Name'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class Name'].values.reshape(-1, 1)).flatten(), name='number')            
    
    elif identifier == 'BIN:Hepatitis':
        feature_names = [
                        'Class',#: DIE, LIVE
                        'AGE',#: 10, 20, 30, 40, 50, 60, 70, 80
                        'SEX',#: male, female
                        'STEROID',#: no, yes
                        'ANTIVIRALS',#: no, yes
                        'FATIGUE',#: no, yes
                        'MALAISE',#: no, yes
                        'ANOREXIA',#: no, yes
                        'LIVER BIG',#: no, yes
                        'LIVER FIRM',#: no, yes
                        'SPLEEN PALPABLE',#: no, yes
                        'SPIDERS',#: no, yes
                        'ASCITES',#: no, yes
                        'VARICES',#: no, yes
                        'BILIRUBIN',#: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
                        #-- see the note below
                        'ALK PHOSPHATE',#: 33, 80, 120, 160, 200, 250
                        'SGOT',#: 13, 100, 200, 300, 400, 500,
                        'ALBUMIN',#: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
                        'PROTIME',#: 10, 20, 30, 40, 50, 60, 70, 80, 90
                        'HISTOLOGY',#: no, yes            
                        ]      
        
        data = pd.read_csv('./real_world_datasets/Hepatitis/hepatitis.data', names=feature_names, index_col=False, na_values='?')
        data = data.fillna(data.mode().iloc[0])
        
        features_select = [
                        'Class',#: DIE, LIVE
                        'AGE',#: 10, 20, 30, 40, 50, 60, 70, 80
                        'SEX',#: male, female
                        'STEROID',#: no, yes
                        'ANTIVIRALS',#: no, yes
                        'FATIGUE',#: no, yes
                        'MALAISE',#: no, yes
                        'ANOREXIA',#: no, yes
                        'LIVER BIG',#: no, yes
                        'LIVER FIRM',#: no, yes
                        'SPLEEN PALPABLE',#: no, yes
                        'SPIDERS',#: no, yes
                        'ASCITES',#: no, yes
                        'VARICES',#: no, yes
                        'BILIRUBIN',#: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
                        #-- see the note below
                        'ALK PHOSPHATE',#: 33, 80, 120, 160, 200, 250
                        'SGOT',#: 13, 100, 200, 300, 400, 500,
                        'ALBUMIN',#: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
                        'PROTIME',#: 10, 20, 30, 40, 50, 60, 70, 80, 90
                        'HISTOLOGY',#: no, yes    
                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
                            'SEX',#: male, female
                            'STEROID',#: no, yes
                            'ANTIVIRALS',#: no, yes
                            'FATIGUE',#: no, yes
                            'MALAISE',#: no, yes
                            'ANOREXIA',#: no, yes
                            'LIVER BIG',#: no, yes
                            'LIVER FIRM',#: no, yes
                            'SPLEEN PALPABLE',#: no, yes
                            'SPIDERS',#: no, yes
                            'ASCITES',#: no, yes
                            'VARICES',#: no, yes
                            'HISTOLOGY',#: no, yes               
                            ]


        X_data = data.drop(['Class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['Class'].values.reshape(-1, 1)).flatten(), name='number')          
        
    elif identifier == 'MULT:Lymphography':
        feature_names = [
                        'class',#: normal find, metastases, malign lymph, fibrosis
                        'lymphatics',#: normal, arched, deformed, displaced
                        'block of affere',#: no, yes
                        'bl. of lymph. c',#: no, yes
                        'bl. of lymph. s',#: no, yes
                        'by pass',#: no, yes
                        'extravasates',#: no, yes
                        'regeneration of',#: no, yes
                        'early uptake in',#: no, yes
                        'lym.nodes dimin',#: 0-3
                        'lym.nodes enlar',#: 1-4
                        'changes in lym.',#: bean, oval, round
                        'efect in node',#: no, lacunar, lac. marginal, lac. central
                        'changes in node',#: no, lacunar, lac. margin, lac. central
                        'changes in stru',#: no, grainy, drop-like, coarse, diluted, reticular, stripped, faint,
                        'special forms',#: no, chalices, vesicles
                        'dislocation of',#: no, yes
                        'exclusion of no',#: no, yes
                        'no. of nodes in',#: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70                    
                        ]      
        
        data = pd.read_csv('./real_world_datasets/Lymphography/lymphography.data', names=feature_names, index_col=False)
        
        features_select = [
                        'class',#: normal find, metastases, malign lymph, fibrosis
                        'lymphatics',#: normal, arched, deformed, displaced
                        'block of affere',#: no, yes
                        'bl. of lymph. c',#: no, yes
                        'bl. of lymph. s',#: no, yes
                        'by pass',#: no, yes
                        'extravasates',#: no, yes
                        'regeneration of',#: no, yes
                        'early uptake in',#: no, yes
                        'lym.nodes dimin',#: 0-3
                        'lym.nodes enlar',#: 1-4
                        'changes in lym.',#: bean, oval, round
                        'efect in node',#: no, lacunar, lac. marginal, lac. central
                        'changes in node',#: no, lacunar, lac. margin, lac. central
                        'changes in stru',#: no, grainy, drop-like, coarse, diluted, reticular, stripped, faint,
                        'special forms',#: no, chalices, vesicles
                        'dislocation of',#: no, yes
                        'exclusion of no',#: no, yes
                        'no. of nodes in',#: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70               
                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
           
                            ]


        X_data = data.drop(['class'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['class'].values.reshape(-1, 1)).flatten(), name='number')          
        
    elif identifier == 'MULT:Zoo':
        feature_names = [
                        'animal name',#: Unique for each instance
                        'hair',#: Boolean
                        'feathers',#: Boolean
                        'eggs',#: Boolean
                        'milk',#: Boolean
                        'airborne',#: Boolean
                        'aquatic',#: Boolean
                        'predator',#: Boolean
                        'toothed',#: Boolean
                        'backbone',#: Boolean
                        'breathes',#: Boolean
                        'venomous',#: Boolean
                        'fins',#: Boolean
                        'legs',#: Numeric (set of values: {0,2,4,5,6,8})
                        'tail',#: Boolean
                        'domestic',#: Boolean
                        'catsize',#: Boolean
                        'type',#: Numeric (integer values in range [1,7])               
                        ]      
        
        data = pd.read_csv('./real_world_datasets/Zoo/zoo.data', names=feature_names, index_col=False)
        
        features_select = [
                        #'animal name',#: Unique for each instance
                        'hair',#: Boolean
                        'feathers',#: Boolean
                        'eggs',#: Boolean
                        'milk',#: Boolean
                        'airborne',#: Boolean
                        'aquatic',#: Boolean
                        'predator',#: Boolean
                        'toothed',#: Boolean
                        'backbone',#: Boolean
                        'breathes',#: Boolean
                        'venomous',#: Boolean
                        'fins',#: Boolean
                        'legs',#: Numeric (set of values: {0,2,4,5,6,8})
                        'tail',#: Boolean
                        'domestic',#: Boolean
                        'catsize',#: Boolean
                        'type',#: Numeric (integer values in range [1,7])                
                        ]  


        data = data[features_select]

        nominal_features = [
                        ]
        ordinal_features = [   
           
                            ]


        X_data = data.drop(['type'], axis = 1)
        y_data = pd.Series(OrdinalEncoder().fit_transform(data['type'].values.reshape(-1, 1)).flatten(), name='number')          
     
    else:
        
        
        raise SystemExit('Unknown key: ' + str(identifier))
    
    ((X_train, y_train),
     (X_valid, y_valid),
     (X_test, y_test),
     normalizer_list) = preprocess_data(X_data, 
                                       y_data,
                                       nominal_features,
                                       ordinal_features,
                                       config,
                                       random_seed=random_seed,
                                       verbosity=verbosity,
                                       normalizer_list = normalizer_list)      

    
    return {
           'X_train': X_train,
           'y_train': y_train,
           'X_valid': X_valid,
           'y_valid': y_valid,
           'X_test': X_test,
           'y_test': y_test,
           'normalizer_list': normalizer_list
           }



def evaluate_gdt(identifier, 
                  random_seed_data=42, 
                  random_seed_model=42, 
                  config=None,
                  benchmark_dict={},
                  metrics=[],
                  hpo=False,
                  verbosity=0):

    
    config_training = deepcopy(config)  
    
    if 'REG:' in identifier:
        config_training['gdt']['objective'] = 'regression'
        if 'loss' not in config_training['gdt']:
            config_training['gdt']['loss'] = 'mse'         
        if 'normalize' not in config_training['gdt']:
            config_training['gdt']['normalize'] = 'mean'           
    elif 'BIN:' in identifier:
        config_training['gdt']['objective'] = 'classification'
        if 'loss' not in config_training['gdt']:
            config_training['gdt']['loss'] = 'crossentropy'     
        config_training['gdt']['normalize'] = None
    elif 'MULT:' in identifier:
        config_training['gdt']['objective'] = 'classification'    
        if 'loss' not in config_training['gdt']:
            config_training['gdt']['loss'] = 'kl_divergence'
        config_training['gdt']['normalize'] = None
         
    if verbosity > 0:
        print('________________________________________________________________________________________________________')   
        
    dataset_dict = {}
    model_dict = {}
    runtime_dict = {}

    scores_dict = {'GDT': {}}
 
    dataset_dict = get_preprocessed_dataset(identifier,
                                            random_seed=random_seed_data,
                                            config=config_training,
                                            verbosity=verbosity)

    
    ##############################################################

    number_of_classes = len(np.unique(np.concatenate([dataset_dict['y_train'].values, dataset_dict['y_valid'].values, dataset_dict['y_test'].values])))

    
    for key, value in benchmark_dict.items():
        scores_dict[key] = {}
        
        if value is None:
            if key == 'sklearn':
                if config_training['computation']['use_best_hpo_result']:
                    
                    if config_training['gdt']['objective'] == 'classification':
                        model_identifier = 'Sklearn_class'
                    else:
                        model_identifier = 'Sklearn_reg'
                    
                    try:
                        hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                              model_identifier=model_identifier, 
                                                                              return_best_only=True, 
                                                                              ascending=False)
                        if verbosity > 1:
                            print('Loaded Sklearn Parameters for ' + identifier + ' with Score ' + str(hpo_results['score']))

                        params = hpo_results['model']
                        
                        if config_training['computation']['force_depth']:
                            params['max_depth'] = config_training['gdt']['depth']
                        
                    except FileNotFoundError:
                        print('No Best Parameters Sklearn for ' + identifier)     
                        params = {'max_depth': config_training['gdt']['depth'],
                                  'random_state': random_seed_model}                        
                    
                else:
                    params = {'max_depth': config_training['gdt']['depth'],
                              'random_state': random_seed_model}
                #print('SKLEARN', params)
                start = timeit.default_timer()

                if config_training['gdt']['objective'] == 'classification':
                     sklearn_model = DecisionTreeClassifier
                else:
                     sklearn_model = DecisionTreeRegressor                   
                
                model = sklearn_model()
                model.set_params(**params)

                X_train_data_extended = pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])
                y_train_data_extended = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])
                
                model.fit(X_train_data_extended, 
                          y_train_data_extended)        

                end = timeit.default_timer()  
                runtime = end - start
            elif key == 'GeneticTree':
                if config_training['computation']['use_best_hpo_result']:
                    try:
                        hpo_results = read_best_hpo_result_from_csv_benchmark(identifier, 
                                                                              model_identifier=key, 
                                                                              return_best_only=True, 
                                                                              ascending=False)
                        if verbosity > 1:
                            print('Loaded GeneticTree Parameters for ' + identifier + ' with Score ' + str(hpo_results['score']))

                        params = hpo_results['model']
                        params['mutation_prob'] = float(params['mutation_prob'])
                        params['mutation_replace'] = bool(params['mutation_replace'])
                        params['cross_prob'] = float(params['cross_prob'])
                        params['cross_both'] = bool(params['cross_both'])
                        
                    except FileNotFoundError:
                        print('No Best Parameters GeneticTree for ' + identifier)          
                        params = {
                                            'n_thresholds': 10,

                                            'n_trees': 400,
                                            'max_iter': 500,
                                            'cross_prob': 0.6,
                                            'mutation_prob': 0.4,

                                            'early_stopping': True,

                                            'max_depth': config_training['gdt']['depth'],
                                            'random_state': config_training['computation']['random_seed'],
                                            'n_jobs': 1
                                        }
                else:                 
                    params = {
                                        'n_thresholds': 10,

                                        'n_trees': 400,
                                        'max_iter': 500,
                                        'cross_prob': 0.6,
                                        'mutation_prob': 0.4,

                                        'early_stopping': True,

                                        'max_depth': config_training['gdt']['depth'],
                                        'random_state': config_training['computation']['random_seed'],
                                        'n_jobs': 1
                                    }
                #print('GENTREE', params)
                start = timeit.default_timer()
                
              #  model = GeneticTree()
                model.set_params(**params)
                
                X_train_data_extended = pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])
                y_train_data_extended = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])
                
                model = model.fit(enforce_numpy(X_train_data_extended), 
                                  enforce_numpy(y_train_data_extended))          

                end = timeit.default_timer()  
                runtime = end - start          
                
            runtime_dict[key] = runtime
            model_dict[key] = model
        else:
            runtime_dict[key] = value[0]
            model_dict[key] = value[1]
            
     
    
    
    ##############################################################

    if hpo:
        if config_training['gdt']['objective'] == 'classification':
            cv_generator = StratifiedKFold(n_splits=config_training['computation']['cv_num'], shuffle=True, random_state=config_training['computation']['random_seed'])
        else:
            cv_generator = KFold(n_splits=config_training['computation']['cv_num'], shuffle=True, random_state=config_training['computation']['random_seed'])        
        
        X_train_valid = pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])
        y_train_valid = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])

        histroy_list = []
        score_base_model_cv_list = {'GDT': {}}

        for model_identifier in model_dict.keys():
            score_base_model_cv_list[model_identifier] = {}
            
        for model_identifier in score_base_model_cv_list.keys():    
            for metric in metrics:
                score_base_model_cv_list[model_identifier][metric] = []
                
        

        for train_index, valid_index in cv_generator.split(X_train_valid, y_train_valid):
            #print("TRAIN:", train_index, "TEST:", test_index)
            if isinstance(X_train_valid, pd.DataFrame) or isinstance(X_train_valid, pd.Series):
                X_train_cv, X_valid_cv = X_train_valid.iloc[train_index], X_train_valid.iloc[valid_index]
            else:
                X_train_cv, X_valid_cv = X_train_valid[train_index], X_train_valid[valid_index]            
            if isinstance(y_train_valid, pd.DataFrame) or isinstance(y_train_valid, pd.Series):
                y_train_cv, y_valid_cv = y_train_valid.iloc[train_index], y_train_valid.iloc[valid_index]
            else:
                y_train_cv, y_valid_cv = y_train_valid[train_index], y_train_valid[valid_index]            

            X_train_cv_train, X_train_cv_valid, y_train_cv_train, y_train_cv_valid = train_test_split(X_train_cv, y_train_cv, test_size=0.1, random_state=config_training['computation']['random_seed'])
            
            if config_training['preprocessing']['normalization_technique'] == 'mean':
                config_training['gdt']['activation'] = None
                
            model_dict['GDT'] = GDT(number_of_variables = X_train_cv_train.shape[1],
                                      number_of_classes = number_of_classes,
                                      
                                      objective = config_training['gdt']['objective'],         
                                    
                                      loss = config_training['gdt']['loss'],

                                      random_seed = random_seed_model,
                                      verbosity = verbosity)  
            
            model_dict['GDT'].set_params(**config_training['gdt'])        
            
            if config_training['computation']['use_gpu']: 
                with tf.device('/device:GPU:0'):
                    history = model_dict['GDT'].fit(X_train_cv_train, 
                                                      y_train_cv_train, 

                                                      batch_size=config_training['gdt']['batch_size'], 
                                                      epochs=config_training['gdt']['epochs'], 

                                                      restarts=config_training['gdt']['restarts'], 
                                                      restart_type=config_training['gdt']['restart_type'], 

                                                      early_stopping_epochs=config_training['gdt']['early_stopping_epochs'], 
                                                      early_stopping_type=config_training['gdt']['early_stopping_type'], 
                                                      early_stopping_epsilon=config_training['gdt']['early_stopping_epsilon'], 

                                                      valid_data=(X_train_cv_valid, y_train_cv_valid))        

            else:
                history = model_dict['GDT'].fit(X_train_cv_train, 
                                                  y_train_cv_train, 

                                                  batch_size=config_training['gdt']['batch_size'], 
                                                  epochs=config_training['gdt']['epochs'], 

                                                  restarts=config_training['gdt']['restarts'], 
                                                  restart_type=config_training['gdt']['restart_type'], 

                                                  early_stopping_epochs=config_training['gdt']['early_stopping_epochs'], 
                                                  early_stopping_type=config_training['gdt']['early_stopping_type'], 
                                                  early_stopping_epsilon=config_training['gdt']['early_stopping_epsilon'], 
                                                
                                                  valid_data=(X_train_cv_valid, y_train_cv_valid))        




            for key in model_dict.keys():
                model_pred = model_dict[key].predict(enforce_numpy(X_valid_cv))
                model_pred = np.nan_to_num(model_pred)

                for metric in metrics:
                    if metric in ['f1', 'accuracy']:
                        model_pred = np.round(model_pred)                        

                    if metric not in ['f1', 'roc_auc']:
                        score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(y_valid_cv, model_pred)
                    else:
                        if metric == 'f1':
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(y_valid_cv, model_pred, average='weighted')
                        elif metric == 'roc_auc':
                            try:
                                if int(np.max(y_train_cv_train)+1) > 2:
                                    score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_cv, num_classes=int(max(pd.concat([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']]))+1)), tf.keras.utils.to_categorical(model_pred, num_classes=int(max(pd.concat([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']]))+1)), multi_class='ovo')
                                else:
                                    #score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_cv), tf.keras.utils.to_categorical(model_pred))
                                    score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(y_valid_cv, model_pred)
                                    
                            except ValueError:
                                score_base_model_cv = 0.5                    
                    score_base_model_cv_list[key][metric].append(score_base_model_cv)
  
        
        if config_training['preprocessing']['normalization_technique'] == 'mean':
            config_training['gdt']['activation'] = None
        
        model_dict['GDT'] = GDT(number_of_variables = dataset_dict['X_train'].shape[1],
                              number_of_classes = number_of_classes,
                                  
                                objective = config_training['gdt']['objective'],
                                  
                                loss = config_training['gdt']['loss'],      
                                  
                                random_seed = random_seed_model,
                                verbosity = verbosity)   
        
        
        model_dict['GDT'].set_params(**config_training['gdt'])        


        start_gdt = timeit.default_timer()

        if config_training['computation']['use_gpu']:
            with tf.device('/device:GPU:0'):             
                scores_dict['history'] = model_dict['GDT'].fit(dataset_dict['X_train'], 
                                                              dataset_dict['y_train'], 
                                                     
                                                              batch_size=config_training['gdt']['batch_size'], 
                                                              epochs=config_training['gdt']['epochs'], 

                                                              restarts=config_training['gdt']['restarts'], 
                                                              restart_type=config_training['gdt']['restart_type'],                                                                 

                                                              early_stopping_epochs=config_training['gdt']['early_stopping_epochs'], 
                                                              early_stopping_type=config_training['gdt']['early_stopping_type'], 
                                                              early_stopping_epsilon=config_training['gdt']['early_stopping_epsilon'], 
                                                               
                                                              valid_data=(dataset_dict['X_valid'], dataset_dict['y_valid']))
        else:
            scores_dict['history'] = model_dict['GDT'].fit(dataset_dict['X_train'], 
                                                          dataset_dict['y_train'], 

                                                          batch_size=config_training['gdt']['batch_size'], 
                                                          epochs=config_training['gdt']['epochs'], 

                                                          restarts=config_training['gdt']['restarts'], 
                                                          restart_type=config_training['gdt']['restart_type'], 
                                                            
                                                          early_stopping_epochs=config_training['gdt']['early_stopping_epochs'], 
                                                          early_stopping_type=config_training['gdt']['early_stopping_type'], 
                                                          early_stopping_epsilon=config_training['gdt']['early_stopping_epsilon'], 
                                                           
                                                          valid_data=(dataset_dict['X_valid'], dataset_dict['y_valid']))            

        end_gdt = timeit.default_timer()
        runtime_dict['GDT'] = end_gdt - start_gdt

        ##############################################################

        for key in model_dict.keys(): 

            if key == 'GDT':
                dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))
                dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))

            else:
                dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])))         
                dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])))         

            dataset_dict['y_test_' + key] = np.nan_to_num(dataset_dict['y_test_' + key])
            dataset_dict['y_valid_' + key] = np.nan_to_num(dataset_dict['y_valid_' + key])
            dataset_dict['y_train_' + key] = np.nan_to_num(dataset_dict['y_train_' + key])                   
                
            if key == 'GDT':          
                y_test_data = dataset_dict['y_test']
                y_valid_data = dataset_dict['y_valid']
                y_train_data = dataset_dict['y_train']
            else:
                y_test_data = dataset_dict['y_test']
                y_valid_data = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])
                y_train_data = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])      


            for metric in metrics:                
                if metric in ['accuracy', 'f1']:
                    y_test = np.round(dataset_dict['y_test_' + key])
                    y_valid = np.round(dataset_dict['y_valid_' + key])  
                    y_train = np.round(dataset_dict['y_train_' + key])  
                else:
                    y_test = dataset_dict['y_test_' + key]
                    y_valid = dataset_dict['y_valid_' + key]               
                    y_train = dataset_dict['y_train_' + key]         

                if metric not in ['f1', 'roc_auc']:
                    scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                    scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                    scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                else:          
                    if metric == 'f1':
                        scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test, average='weighted')
                        scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid, average='weighted')
                        scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train, average='weighted')
                    elif metric == 'roc_auc':
                        try:
                            if int(np.max(y_train_data)+1) > 2:                            
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_test, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                            else:
                                #scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data), tf.keras.utils.to_categorical(y_test))
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                        except ValueError:
                            scores_dict[key][metric + '_test'] = 0.5

                        try:
                            if int(np.max(y_train_data)+1) > 2:                            
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_valid, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                            else:
                                #scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data), tf.keras.utils.to_categorical(y_valid))
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                        except ValueError:
                            scores_dict[key][metric + '_valid'] = 0.5 

                        try:
                            if int(np.max(y_train_data)+1) > 2:                            
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_train_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_train, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                            else:
                                #scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data), tf.keras.utils.to_categorical(y_valid))
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                        except ValueError:
                            scores_dict[key][metric + '_train'] = 0.5    

                if verbosity > 0:
                    print('Test ' + metric + ' ' + key + ' (' + str(0) + ')', scores_dict[key][metric + '_test'])

            if verbosity > 0:
                print('________________________________________________________________________________________________________')            

            
            
            scores_dict[key]['runtime'] = runtime_dict[key]  


          
    else:  
            
        if config_training['computation']['use_best_hpo_result']:
            try:
                best_params = read_best_hpo_result_from_csv(identifier, 
                                              counter='', 
                                              return_best_only=True, 
                                              ascending=False)
                if verbosity > 1:
                    print('Loaded Parameters GDT for ' + identifier + ' with Score ' + str(best_params['score']))
                #print('best_params', best_params)
                
                
                for model_param_key, model_param_value in best_params['model'].items():
                    if model_param_key == 'depth' and config_training['computation']['force_depth']:
                        depth_value = config_training['gdt']['depth']
                        if verbosity > 0:
                            print('Setting depth to ' + str(depth_value))
                        config_training['gdt']['depth'] = depth_value     
                    elif model_param_key == 'dropout':
                        dropout_value = config_training['gdt']['dropout']
                        if verbosity > 0:
                            print('Setting dropout to ' + str(dropout_value))
                        config_training['gdt']['dropout'] =dropout_value                                                     
                    else:
                        config_training['gdt'][model_param_key] = model_param_value
                    
                for training_param_key, training_param_value in best_params['training'].items():
                    if training_param_key == 'restarts':
                        restart_value = config_training['gdt']['restarts']#max(training_param_value, config_training['gdt']['restarts'])
                        if verbosity > 1:
                            print('Setting restarts to ' + str(restart_value))
                        config_training['gdt']['restarts'] = restart_value
                    elif training_param_key == 'restart_type':
                        restart_type_value = config_training['gdt']['restart_type']
                        if verbosity > 1:
                            print('Setting restart_type to ' + str(restart_type_value))
                        config_training['gdt']['restart_type'] = restart_type_value  
                    elif training_param_key == 'early_stopping_epochs':
                        early_stopping_epochs_value = max(training_param_value, config_training['gdt']['early_stopping_epochs'])
                        if verbosity > 1:
                            print('Setting early_stopping_epochs to ' + str(early_stopping_epochs_value))
                        config_training['gdt']['early_stopping_epochs'] = early_stopping_epochs_value
                    elif training_param_key == 'early_stopping_type':
                        early_stopping_type_value = config_training['gdt']['early_stopping_type']
                        if verbosity > 1:
                            print('Setting early_stopping_type to ' + str(early_stopping_type_value))
                        config_training['gdt']['early_stopping_type'] = early_stopping_type_value
                    elif training_param_key == 'early_stopping_epsilon':
                        early_stopping_epsilon_value = config_training['gdt']['early_stopping_epsilon']
                        if verbosity > 1:
                            print('Setting early_stopping_epsilon to ' + str(early_stopping_epsilon_value))
                        config_training['gdt']['early_stopping_epsilon'] = early_stopping_epsilon_value           
                    elif training_param_key == 'epochs':
                        epochs_value = config_training['gdt']['epochs']#max(training_param_value, config_training['gdt']['epochs'])
                        if verbosity > 1:
                            print('Setting epochs to ' + str(epochs_value))
                        config_training['gdt']['epochs'] = epochs_value         
                    elif training_param_key == 'batch_size':
                        batch_size_value = config_training['gdt']['batch_size']
                        if verbosity > 1:
                            print('Setting batch_size to ' + str(batch_size_value))
                        config_training['gdt']['batch_size'] = batch_size_value                            
                    else:
                        config_training['gdt'][training_param_key] = training_param_value
                                
            except FileNotFoundError:
                print('No Best Parameters GDT for ' + identifier)
            
         
        if config_training['preprocessing']['normalization_technique'] == 'mean':
            config_training['gdt']['activation'] = None
            
        model_dict['GDT'] = GDT(number_of_variables = dataset_dict['X_train'].shape[1],
                              number_of_classes = number_of_classes,
                                  
                                objective = config_training['gdt']['objective'],
                                  
                                loss = config_training['gdt']['loss'], 
                                  
                                random_seed = random_seed_model,
                                verbosity = verbosity)   
        
        model_dict['GDT'].set_params(**config_training['gdt'])        


        start_gdt = timeit.default_timer()

        
        if config_training['computation']['use_gpu']:
            with tf.device('/device:GPU:0'):           
                scores_dict['history'] = model_dict['GDT'].fit(dataset_dict['X_train'], 
                                                              dataset_dict['y_train'], 
                                                     
                                                              batch_size=config_training['gdt']['batch_size'], 
                                                              epochs=config_training['gdt']['epochs'], 

                                                              restarts=config_training['gdt']['restarts'], 
                                                              restart_type=config_training['gdt']['restart_type'],                                                                 

                                                              early_stopping_epochs=config_training['gdt']['early_stopping_epochs'], 
                                                              early_stopping_type=config_training['gdt']['early_stopping_type'], 
                                                              early_stopping_epsilon=config_training['gdt']['early_stopping_epsilon'], 
                                                               
                                                              valid_data=(dataset_dict['X_valid'], dataset_dict['y_valid']))
        else:
            scores_dict['history'] = model_dict['GDT'].fit(dataset_dict['X_train'], 
                                                          dataset_dict['y_train'], 

                                                          batch_size=config_training['gdt']['batch_size'], 
                                                          epochs=config_training['gdt']['epochs'], 

                                                          restarts=config_training['gdt']['restarts'], 
                                                          restart_type=config_training['gdt']['restart_type'],                                                             

                                                          early_stopping_epochs=config_training['gdt']['early_stopping_epochs'], 
                                                          early_stopping_type=config_training['gdt']['early_stopping_type'], 
                                                          early_stopping_epsilon=config_training['gdt']['early_stopping_epsilon'], 
                                                           
                                                          valid_data=(dataset_dict['X_valid'], dataset_dict['y_valid']))
                
        end_gdt = timeit.default_timer()
        runtime_dict['GDT'] = end_gdt - start_gdt

        ##############################################################

        for key in model_dict.keys(): 

            if key == 'GDT':
                dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))
                dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))

            else:
                dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
                dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])))         
                dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])))         

            dataset_dict['y_test_' + key] = np.nan_to_num(dataset_dict['y_test_' + key])
            dataset_dict['y_valid_' + key] = np.nan_to_num(dataset_dict['y_valid_' + key])
            dataset_dict['y_train_' + key] = np.nan_to_num(dataset_dict['y_train_' + key])                   
                
            if key == 'GDT':          
                y_test_data = dataset_dict['y_test']
                y_valid_data = dataset_dict['y_valid']
                y_train_data = dataset_dict['y_train']
            else:
                y_test_data = dataset_dict['y_test']
                y_valid_data = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])
                y_train_data = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])      


            for metric in metrics:                
                if metric in ['accuracy', 'f1']:
                    y_test = np.round(dataset_dict['y_test_' + key])
                    y_valid = np.round(dataset_dict['y_valid_' + key])  
                    y_train = np.round(dataset_dict['y_train_' + key])  
                else:
                    y_test = dataset_dict['y_test_' + key]
                    y_valid = dataset_dict['y_valid_' + key]               
                    y_train = dataset_dict['y_train_' + key]         

                if metric not in ['f1', 'roc_auc']:
                    scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                    scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                    scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                else:          
                    if metric == 'f1':
                        scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test, average='weighted')
                        scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid, average='weighted')
                        scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train, average='weighted')
                    elif metric == 'roc_auc':
                        try:
                            if int(np.max(y_train_data)+1) > 2:                            
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_test, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                            else:
                                #scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data), tf.keras.utils.to_categorical(y_test))
                                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                        except ValueError:
                            scores_dict[key][metric + '_test'] = 0.5  

                        try:
                            if int(np.max(y_train_data)+1) > 2:                            
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_valid, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                            else:
                                #scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data), tf.keras.utils.to_categorical(y_valid))
                                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                        except ValueError:
                            scores_dict[key][metric + '_valid'] = 0.5       

                        try:
                            if int(np.max(y_train_data)+1) > 2:                            
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_train_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_train, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                            else:
                                #scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data), tf.keras.utils.to_categorical(y_valid))
                                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                        except ValueError:
                            scores_dict[key][metric + '_train'] = 0.5 

                if verbosity > 0:
                    print('Test ' + metric + ' ' + key + ' (' + str(0) + ')', scores_dict[key][metric + '_test'])

            if verbosity > 0:
                print('________________________________________________________________________________________________________') 
                
                
            scores_dict[key]['runtime'] = runtime_dict[key]  
            
    return identifier, dataset_dict, model_dict, scores_dict
    

def evaluate_real_world_parallel_nested(identifier_list, 
                                  random_seed_data=42, 
                                  random_seed_model=42, 
                                  config=None,
                                  benchmark_dict={},
                                  metrics=[],
                                  verbosity=0):
    
    dataset_dict = {}
    model_dict = {}

    scores_dict = {}
        
    parallel_eval_real_world_nested = Parallel(n_jobs=max(1, min(config['computation']['n_jobs']//config['computation']['trials'], len(identifier_list))), verbose=3, backend='loky') #loky #sequential multiprocessing
    
    
    evaluation_results_real_world_nested = parallel_eval_real_world_nested(delayed(evaluate_gdt)(identifier,
                                                                                                  random_seed_data=random_seed_data, 
                                                                                                  random_seed_model=random_seed_model, 
                                                                                                  config=config,
                                                                                                  benchmark_dict=benchmark_dict,
                                                                                                  metrics=metrics,
                                                                                                  verbosity=verbosity) for identifier in identifier_list)    
    
    
    for (identifier, dataset_dict_entry, model_dict_entry, scores_dict_entry) in evaluation_results_real_world_nested:
        dataset_dict[identifier] = dataset_dict_entry
        model_dict[identifier] = model_dict_entry
        scores_dict[identifier] = scores_dict_entry

        
    return model_dict, scores_dict, dataset_dict


def evaluate_real_world_parallel(identifier_list, 
                                  random_seed_data=42, 
                                  random_seed_model=42, 
                                  config=None,
                                  benchmark_dict={},
                                  metrics=[],
                                  hpo=False,
                                  verbosity=0):

    dataset_dict = {}
    model_dict = {}

    scores_dict = {}
    
    disable = True if verbosity <= 0 else False
    for identifier in tqdm(identifier_list, desc='dataset loop', disable=disable):

        (identifier,
         dataset_dict[identifier], 
         model_dict[identifier], 
         scores_dict[identifier]) = evaluate_gdt(identifier, 
                                                  random_seed_data=random_seed_data, 
                                                  random_seed_model=random_seed_model, 
                                                  config=config,
                                                  benchmark_dict=benchmark_dict,
                                                  metrics=metrics,
                                                  hpo=hpo,
                                                  verbosity=verbosity)
        
    return model_dict, scores_dict, dataset_dict


def evaluate_parameter_setting_real_world(parameter_setting, 
                                          identifier, 
                                          config,
                                          benchmark_dict={},
                                          metrics=[],
                                          hpo=False):
    
    config_parameter_setting = deepcopy(config)
    
    
    for key, value in parameter_setting.items():
        config_parameter_setting['gdt'][key] = value
    
    
    evaluation_results_real_world = []
    for i in range(config['computation']['trials']):
        evaluation_result = evaluate_real_world_parallel(identifier_list=[identifier], 
                                                           random_seed_data=config['computation']['random_seed']+i,
                                                           random_seed_model=config['computation']['random_seed'],
                                                           config = config_parameter_setting,
                                                           benchmark_dict=benchmark_dict,
                                                           metrics = metrics,
                                                           hpo=hpo,
                                                           verbosity = -1)
        evaluation_results_real_world.append(evaluation_result)
        
    del evaluation_result

    for i, real_world_result in enumerate(evaluation_results_real_world):
        if i == 0:
            model_dict_real_world = real_world_result[0]
            scores_dict_real_world = real_world_result[1]
            dataset_dict_real_world = real_world_result[2]
        else: 
            model_dict_real_world = mergeDict(model_dict_real_world, real_world_result[0])
            scores_dict_real_world = mergeDict(scores_dict_real_world, real_world_result[1])
            dataset_dict_real_world = mergeDict(dataset_dict_real_world, real_world_result[2])    

    del real_world_result, evaluation_results_real_world

    
    model_identifier_list = flatten_list(['GDT', list(benchmark_dict.keys())])           
    
    metric_identifer_list = ['_valid', '_test']

    index = [identifier]
    columns = flatten_list([[[approach + ' ' + metric + '_mean', 
                              approach + ' ' + metric + '_max', 
                              approach + ' ' + metric + '_std', 
                              approach + ' mean runtime'] for metric in metrics] for approach in model_identifier_list])



    
    
    scores_dataframe_real_world_dict = {}
    
    for metric_identifer in metric_identifer_list:
        results_list = []
        for model_identifier in model_identifier_list:
            results = None

            for metric in metrics:
                scores = [scores_dict_real_world[identifier][model_identifier][metric + metric_identifer] for identifier in [identifier]]
                scores_mean = np.mean(scores, axis=1) if config['computation']['trials'] > 1 else scores
                scores_max = np.max(scores, axis=1) if config['computation']['trials'] > 1 else scores
                scores_std = np.std(scores, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * len(scores))

                runtimes = np.array([scores_dict_real_world[identifier][model_identifier]['runtime'] for identifier in [identifier]])
                runtimes_mean = np.mean(runtimes, axis=1) if config['computation']['trials'] > 1 else runtimes
                
                results_by_metric = np.vstack([scores_mean, scores_max, scores_std, runtimes_mean])

                if results is None:
                    results = results_by_metric        
                else:
                    results = np.vstack([results, results_by_metric])             

            results_list.append(results)
            
        scores_dataframe_real_world = pd.DataFrame(data=np.vstack(results_list).T, index = index, columns = columns)   
        scores_dataframe_real_world_dict[metric_identifer[1:]] = scores_dataframe_real_world   

    del scores_dict_real_world, dataset_dict_real_world
    
    model_params = model_dict_real_world[identifier]['GDT'][0].get_params() if isinstance(model_dict_real_world[identifier]['GDT'], list) else model_dict_real_world[identifier]['GDT'].get_params()
    
    return scores_dataframe_real_world_dict, parameter_setting, model_dict_real_world, model_params

    
    
def sleep_minutes(minutes):  
    if minutes > 0:
        for _ in tqdm(range(minutes)):
            time.sleep(60)
            
            



def generate_GDT_from_config(number_of_variables,
                              number_of_classes,
                              verbosity,
                              config,
                              identifier=None):
    
    config_training = deepcopy(config)
    
    if config_training['computation']['use_best_hpo_result']:
        try:
            best_params = read_best_hpo_result_from_csv(identifier, 
                                          counter='', 
                                          return_best_only=True, 
                                          ascending=False)
            if verbosity > 1:
                print('Loaded Parameters GDT for ' + identifier + ' with Score ' + str(best_params['score']))
            #print('best_params', best_params)


            for model_param_key, model_param_value in best_params['model'].items():
                if model_param_key == 'depth' and config_training['computation']['force_depth']:
                    depth_value = config_training['gdt']['depth']
                    if verbosity > 0:
                        print('Setting depth to ' + str(depth_value))
                    config_training['gdt']['depth'] = depth_value    
                elif model_param_key == 'dropout':
                    dropout_value = config_training['gdt']['dropout']
                    if verbosity > 0:
                        print('Setting dropout to ' + str(dropout_value))
                    config_training['gdt']['dropout'] = dropout_value                                      
                else:
                    config_training['gdt'][model_param_key] = model_param_value

            for training_param_key, training_param_value in best_params['training'].items():
                if training_param_key == 'restarts':
                    restart_value = config_training['gdt']['restarts']#max(training_param_value, config_training['gdt']['restarts'])
                    if verbosity > 1:
                        print('Setting restarts to ' + str(restart_value))
                    config_training['gdt']['restarts'] = restart_value
                elif training_param_key == 'restart_type':
                    restart_type_value = config_training['gdt']['restart_type']
                    if verbosity > 1:
                        print('Setting restart_type to ' + str(restart_type_value))
                    config_training['gdt']['restart_type'] = restart_type_value  
                elif training_param_key == 'early_stopping_epochs':
                    early_stopping_epochs_value = max(training_param_value, config_training['gdt']['early_stopping_epochs'])
                    if verbosity > 1:
                        print('Setting early_stopping_epochs to ' + str(early_stopping_epochs_value))
                    config_training['gdt']['early_stopping_epochs'] = early_stopping_epochs_value
                elif training_param_key == 'early_stopping_type':
                    early_stopping_type_value = config_training['gdt']['early_stopping_type']
                    if verbosity > 1:
                        print('Setting early_stopping_type to ' + str(early_stopping_type_value))
                    config_training['gdt']['early_stopping_type'] = early_stopping_type_value
                    if training_param_key == 'early_stopping_epsilon':
                        early_stopping_epsilon_value = config_training['gdt']['early_stopping_epsilon']
                        if verbosity > 1:
                            print('Setting early_stopping_epsilon to ' + str(early_stopping_epsilon_value))
                        config_training['gdt']['early_stopping_epsilon'] = early_stopping_epsilon_value
                elif training_param_key == 'epochs':
                    epochs_value = config_training['gdt']['epochs']#max(training_param_value, config_training['gdt']['epochs'])
                    if verbosity > 1:
                        print('Setting epochs to ' + str(epochs_value))
                    config_training['gdt']['epochs'] = epochs_value         
                elif training_param_key == 'batch_size':
                    batch_size_value = config_training['gdt']['batch_size']
                    if verbosity > 1:
                        print('Setting batch_size to ' + str(batch_size_value))
                    config_training['gdt']['batch_size'] = batch_size_value                            
                else:
                    config_training['gdt'][training_param_key] = training_param_value

        except FileNotFoundError:
            print('No Best Parameters GDT for ' + identifier)

    if config_training['preprocessing']['normalization_technique'] == 'mean':
        config_training['gdt']['activation'] = None
                
    
    model = GDT(number_of_variables = number_of_variables,
                  number_of_classes = number_of_classes,
                
                 objective = config_training['gdt']['objective'],
                 
                 
                    loss = config_training['gdt']['loss'],

                    random_seed = config_training['computation']['random_seed'],
                    verbosity = verbosity)#5      
        
    model.set_params(**config_training['gdt'])        
    
    return model
    
    
def calculate_scores(model_dict, 
                     dataset_dict,
                     scores_dict = {},
                     metrics = [],
                     verbosity = 1): 

    for key in model_dict.keys(): 

        if key == 'GDT':
            dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
            dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_valid']))
            dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_train']))
            
        else:
            dataset_dict['y_test_' + key] = model_dict[key].predict(enforce_numpy(dataset_dict['X_test']))
            dataset_dict['y_valid_' + key] = model_dict[key].predict(enforce_numpy(pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])))         
            dataset_dict['y_train_' + key] = model_dict[key].predict(enforce_numpy(pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])))      
            
        dataset_dict['y_test_' + key] = np.nan_to_num(dataset_dict['y_test_' + key])
        dataset_dict['y_valid_' + key] = np.nan_to_num(dataset_dict['y_valid_' + key])
        dataset_dict['y_train_' + key] = np.nan_to_num(dataset_dict['y_train_' + key])               
        
        if key == 'GDT':          
            y_test_data = dataset_dict['y_test']
            y_valid_data = dataset_dict['y_valid']
            y_train_data = dataset_dict['y_train']
        else:
            y_test_data = dataset_dict['y_test']
            y_valid_data = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])
            y_train_data = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])      
            
            
        for metric in metrics:                
            if metric in ['accuracy', 'f1']:
                y_test = np.round(dataset_dict['y_test_' + key])
                y_valid = np.round(dataset_dict['y_valid_' + key])  
                y_train = np.round(dataset_dict['y_train_' + key])  
            else:
                y_test = dataset_dict['y_test_' + key]
                y_valid = dataset_dict['y_valid_' + key]               
                y_train = dataset_dict['y_train_' + key]         
                
            if metric not in ['f1', 'roc_auc']:
                scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
            else:          
                if metric == 'f1':
                    scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test, average='weighted')
                    scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid, average='weighted')
                    scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train, average='weighted')
                elif metric == 'roc_auc':
                    try:
                        if int(np.max(y_train_data)+1) > 2:                            
                            scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_test, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                        else:
                            #scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_test_data), tf.keras.utils.to_categorical(y_test))
                            scores_dict[key][metric + '_test'] = sklearn.metrics.get_scorer(metric)._score_func(y_test_data, y_test)
                    except ValueError:
                        scores_dict[key][metric + '_test'] = 0.5

                    try:
                        if int(np.max(y_train_data)+1) > 2:                            
                            scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_valid, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                        else:
                            #scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data), tf.keras.utils.to_categorical(y_valid))
                            scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(y_valid_data, y_valid)
                    except ValueError:
                        scores_dict[key][metric + '_valid'] = 0.5

                    try:
                        if int(np.max(y_train_data)+1) > 2:                            
                            scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_train_data, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), tf.keras.utils.to_categorical(y_train, num_classes=int(max(pd.concat([y_train_data, y_valid_data, y_test_data]))+1)), multi_class='ovo')
                        else:
                            #scores_dict[key][metric + '_valid'] = sklearn.metrics.get_scorer(metric)._score_func(tf.keras.utils.to_categorical(y_valid_data), tf.keras.utils.to_categorical(y_valid))
                            scores_dict[key][metric + '_train'] = sklearn.metrics.get_scorer(metric)._score_func(y_train_data, y_train)
                    except ValueError:
                        scores_dict[key][metric + '_train'] = 0.5

            if verbosity > 0:
                print('Test ' + metric + ' ' + key + ' (' + str(0) + ')', scores_dict[key][metric + '_test'])

        if verbosity > 0:
            print('________________________________________________________________________________________________________')   


# Read idx file format (from LibSVM)   https://github.com/StephanLorenzen/MajorityVoteBounds/blob/278a2811774e48093a7593e068e5958832cfa686/mvb/data.py
def _read_idx_file(path, d, sep=None):
    X = []
    Y = []
    with open(path) as f:
        for l in f:
            x = np.zeros(d)
            l = l.strip().split() if sep is None else l.strip().split(sep)
            Y.append(int(l[0]))
            for pair in l[1:]:
                pair = pair.strip()
                if pair=='':
                    continue
                (i,v) = pair.split(":")
                if v=='':
                    import pdb; pdb.set_trace()
                x[int(i)-1] = float(v)
            X.append(x)
    return np.array(X),np.array(Y)

def enforce_numpy(data):
    
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data_numpy = data.values
    else:
        data_numpy = data
        
    return data_numpy

def get_columns_by_name(df, columnname):
    columns = list(df.columns)
    columns_slected = [name for name in columns if columnname == ' '.join(name.split(' ')[1:])]
    return df[columns_slected]

def one_hot_encode_data(df, transformer=None):
    
    if transformer is None:
        transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), df.columns)], remainder='passthrough', sparse_threshold=0)
        transformer.fit(df)        

        df_values = transformer.transform(df)
        df = pd.DataFrame(df_values, columns=transformer.get_feature_names())
        
        return df, transformer
    else:
        df_values = transformer.transform(df)
        df = pd.DataFrame(df_values, columns=transformer.get_feature_names())
        
    return df


def structure_hpo_results_complete(model_identifier_list,
                                   hpo_results_valid, 
                                   hpo_results_unsorted_valid, 
                                   hpo_results_test, 
                                   hpo_results_unsorted_test,
                                   metrics):
    
    results_valid = []
    columns_valid = []
    for model_identifier in model_identifier_list:
        if model_identifier == 'GDT':
            results_valid.append([hpo_results_valid[key][0]['GDT mean (mean)'] for key in hpo_results_valid.keys()])
        else:
            results_valid.append([hpo_results_valid[key][0][model_identifier + ' mean'] for key in hpo_results_valid.keys()])
        columns_valid.append(model_identifier + ' (mean)')
        
        results_valid.append([hpo_results_valid[key][0][model_identifier + ' runtime mean'] for key in hpo_results_valid.keys()])
        columns_valid.append(model_identifier + ' runtime (mean)')

        
        
    #######################################################################################################################################
    best_results_valid = pd.DataFrame(data=np.vstack(results_valid).T, index=list(hpo_results_valid.keys()), columns=columns_valid)
    best_results_mean_valid = best_results_valid.mean()
    best_results_mean_valid.name = 'MEAN'
    
    count_list = []
    for column in best_results_valid.columns:
        column_metric_identifier = ' '.join(column.split(' ')[1:])
        count_series = best_results_valid[column]>=get_columns_by_name(best_results_valid, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    best_results_count_valid = pd.Series(count_list, index = best_results_valid.columns)
    best_results_count_valid.name = 'COUNT'   
    
    best_results_valid = best_results_valid.append(best_results_mean_valid)
    best_results_valid = best_results_valid.append(best_results_count_valid)
    
    #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
    #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
    reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(2)])
    #print('best_results_valid', best_results_valid.shape, best_results_valid)
    
    best_results_valid = best_results_valid.iloc[:,reorder]
    
    #######################################################################################################################################
        
    results_test = []
    columns_test = []
    for model_identifier in model_identifier_list:
        if model_identifier == 'GDT':
            results_test.append([hpo_results_test[key][0]['GDT mean (mean)'] for key in hpo_results_test.keys()])
        else:
            results_test.append([hpo_results_test[key][0][model_identifier + ' mean'] for key in hpo_results_test.keys()])
        columns_test.append(model_identifier + ' (mean)')
        
        results_test.append([hpo_results_test[key][0][model_identifier + ' runtime mean'] for key in hpo_results_test.keys()])
        columns_test.append(model_identifier + ' runtime (mean)')   
        
    best_results_test = pd.DataFrame(data=np.vstack([results_test]).T, index=list(hpo_results_test.keys()), columns=columns_test)
    best_results_mean_test = best_results_test.mean()
    best_results_mean_test.name = 'MEAN'

    count_list = []
    for column in best_results_test.columns:
        column_metric_identifier = ' '.join(column.split(' ')[1:])
        count_series = best_results_test[column]>=get_columns_by_name(best_results_test, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    best_results_count_test = pd.Series(count_list, index = best_results_test.columns)
    best_results_count_test.name = 'COUNT' 
    
    best_results_test = best_results_test.append(best_results_mean_test)    
    best_results_test = best_results_test.append(best_results_count_test)    

    #reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
    #reorder = flatten_list([[i*len(metrics)+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
    reorder = flatten_list([[i*2+j for i in range(len(model_identifier_list))] for j in range(2)])
    best_results_test = best_results_test.iloc[:,reorder]
    
    return best_results_valid, best_results_test
    



def structure_hpo_results_for_dataset(evaluation_results_hpo, 
                                      model_identifier_list, 
                                      hpo_results_valid, 
                                      hpo_results_unsorted_valid, 
                                      hpo_results_test, 
                                      hpo_results_unsorted_test,                                    
                                      comparator_metric='f1', 
                                      greater_better=True,
                                      config=None,
                                      identifier=None):

    mean_list_unsorted_valid = {}
    mean_list_valid = {}
    mean_list_unsorted_test = {}
    mean_list_test = {}

    runtime_dict_unsorted = {}
    runtime_dict = {}
    for model_identifer in model_identifier_list:

        if model_identifer == 'GDT':
            mean_list_unsorted_valid[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(config['computation']['search_iterations'])]
            mean_list_valid[model_identifer] = sorted(mean_list_unsorted_valid[model_identifer], reverse=greater_better)
            mean_list_unsorted_test[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['test'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(config['computation']['search_iterations'])]
            mean_list_test[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_valid[model_identifer], mean_list_unsorted_test[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]

            mean_list_unsorted_valid[model_identifer + '_max'] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' '  + comparator_metric + '_max']) for i in range(config['computation']['search_iterations'])]
            mean_list_valid[model_identifer + '_max'] = [x for (y,x) in sorted(zip(mean_list_unsorted_valid[model_identifer], mean_list_unsorted_valid[model_identifer + '_max']), key=lambda pair: pair[0], reverse=greater_better)]
            mean_list_unsorted_test[model_identifer + '_max'] = [np.mean(evaluation_results_hpo[i][0]['test'][model_identifer + ' '  + comparator_metric + '_max']) for i in range(config['computation']['search_iterations'])]
            mean_list_test[model_identifer + '_max'] = [x for (y,x) in sorted(zip(mean_list_unsorted_valid[model_identifer], mean_list_unsorted_test[model_identifer + '_max']), key=lambda pair: pair[0], reverse=greater_better)]

        else:
            mean_list_unsorted_valid[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(config['computation']['search_iterations'])]
            mean_list_valid[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_valid['GDT'], mean_list_unsorted_valid[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]
            mean_list_unsorted_test[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['test'][model_identifer + ' ' + comparator_metric + '_mean']) for i in range(config['computation']['search_iterations'])]
            mean_list_test[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_valid['GDT'], mean_list_unsorted_test[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]


        runtime_dict_unsorted[model_identifer] = [np.mean(evaluation_results_hpo[i][0]['valid'][model_identifer + ' mean runtime'].iloc[:,1]) for i in range(config['computation']['search_iterations'])]
        runtime_dict[model_identifer] = [x for (y,x) in sorted(zip(mean_list_unsorted_valid['GDT'], runtime_dict_unsorted[model_identifer]), key=lambda pair: pair[0], reverse=greater_better)]  

    parameter_setting_list_unsorted = [evaluation_results_hpo[i][1] for i in range(config['computation']['search_iterations'])]
    parameter_setting_list = [x for (y,x) in sorted(zip(mean_list_unsorted_valid['GDT'], parameter_setting_list_unsorted), key=lambda pair: pair[0], reverse=greater_better)]

    parameter_setting_list_complete_unsorted = [evaluation_results_hpo[i][3] for i in range(config['computation']['search_iterations'])]#[list(evaluation_results_hpo[i][2].values())[0]['GDT'][0].get_params() for i in range(config['computation']['search_iterations'])]
    parameter_setting_list_complete = [x for (y,x) in sorted(zip(mean_list_unsorted_valid['GDT'], parameter_setting_list_complete_unsorted), key=lambda pair: pair[0], reverse=greater_better)]      
    
    hpo_results_by_identifer = []
    for i in range(config['computation']['search_iterations']):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GDT':
                result_dict['GDT mean (mean)'] = mean_list_valid['GDT'][i]
                result_dict['GDT max (mean)'] = mean_list_valid['GDT_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_valid[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete[i]

        hpo_results_by_identifer.append(result_dict)
    hpo_results_valid[identifier] = hpo_results_by_identifer


    hpo_results_by_identifer_unsorted = []
    for i in range(config['computation']['search_iterations']):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GDT':
                result_dict['GDT mean (mean)'] = mean_list_unsorted_valid['GDT'][i]
                result_dict['GDT max (mean)'] = mean_list_unsorted_valid['GDT_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_unsorted_valid[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict_unsorted[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list_unsorted[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete_unsorted[i]

        hpo_results_by_identifer_unsorted.append(result_dict)
    hpo_results_unsorted_valid[identifier] = hpo_results_by_identifer_unsorted      



    ############
    hpo_results_by_identifer = []
    for i in range(config['computation']['search_iterations']):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GDT':
                result_dict['GDT mean (mean)'] = mean_list_test['GDT'][i]
                result_dict['GDT max (mean)'] = mean_list_test['GDT_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_test[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete[i]

        hpo_results_by_identifer.append(result_dict)
    hpo_results_test[identifier] = hpo_results_by_identifer


    hpo_results_by_identifer_unsorted = []
    for i in range(config['computation']['search_iterations']):
        result_dict = {}
        for model_identifier in model_identifier_list:
            if model_identifier == 'GDT':
                result_dict['GDT mean (mean)'] = mean_list_unsorted_test['GDT'][i]
                result_dict['GDT max (mean)'] = mean_list_unsorted_test['GDT_max'][i]
            else:
                result_dict[model_identifier + ' mean'] = mean_list_unsorted_test[model_identifier][i]

            result_dict[model_identifier + ' runtime mean'] = runtime_dict_unsorted[model_identifier][i]
        result_dict['parameters'] = parameter_setting_list_unsorted[i]
        result_dict['parameters_complete'] = parameter_setting_list_complete_unsorted[i]

        hpo_results_by_identifer_unsorted.append(result_dict)
    hpo_results_unsorted_test[identifier] = hpo_results_by_identifer_unsorted

    print('')
    #display(hpo_results_by_identifer[:1])
    print('___________________________________________________________________________')

    return hpo_results_valid, hpo_results_unsorted_valid, hpo_results_test, hpo_results_unsorted_test
   

def get_params_gentree(tree, config):
    
    return {
                 'n_thresholds': tree._n_thresholds,
                 'n_trees': tree.selector.n_trees,
                 'max_iter': tree.stopper.max_iter,
                 'cross_prob': tree.crosser.cross_prob,
                 'mutation_prob': tree.mutator.mutation_prob,
                 ###'initialization': tree.initializer.initialization,
                 ###'metric': tree.evaluator.metric,
                 ###'selection': tree.selector.selection,
                 'n_elitism': tree.selector.n_elitism,
                 'early_stopping': tree.stopper.early_stopping,
                 'n_iter_no_change': tree.stopper.n_iter_no_change,

                 # additional genetic algorithm params
                 'cross_both': tree.crosser.cross_both,
                 ###'mutations_additional': tree.mutator.mutations_additional,
                 'mutation_replace': tree.mutator.mutation_replace,
                 'initial_depth': tree.initializer.initial_depth,
                 'split_prob': tree.initializer.split_prob,
                 ###'n_leaves_factor': tree.n_leaves_factor,
                 ###'depth_factor': tree.depth_factor,
                 ###'tournament_size': tree.tournament_size,
                 ###'leave_selected_parents': tree.leave_selected_parents,

                 # technical params
                 'random_state': config['computation']['random_seed'],  
    
    }
    
    

    
    
def hpo_gentree(identifier, dataset_dict, parameter_dict, config, metric='f1', greater_better=True):
    
    def evaluate_parameter_setting_gen(parameter_setting, dataset_dict, config):    
        if config['gdt']['objective'] == 'classification':
            cv_generator = StratifiedKFold(n_splits=config['computation']['cv_num'], shuffle=True, random_state=config['computation']['random_seed'])
        else:
            cv_generator = KFold(n_splits=config['computation']['cv_num'], shuffle=True, random_state=config['computation']['random_seed'])        

        X_train_valid = pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])
        y_train_valid = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])

        score_base_model_cv_list = []
        for train_index, valid_index in cv_generator.split(X_train_valid, y_train_valid):
            #print("TRAIN:", train_index, "TEST:", test_index)
            if isinstance(X_train_valid, pd.DataFrame) or isinstance(X_train_valid, pd.Series):
                X_train_cv, X_valid_cv = X_train_valid.iloc[train_index], X_train_valid.iloc[valid_index]
            else:
                X_train_cv, X_valid_cv = X_train_valid[train_index], X_train_valid[valid_index]            
            if isinstance(y_train_valid, pd.DataFrame) or isinstance(y_train_valid, pd.Series):
                y_train_cv, y_valid_cv = y_train_valid.iloc[train_index], y_train_valid.iloc[valid_index]
            else:
                y_train_cv, y_valid_cv = y_train_valid[train_index], y_train_valid[valid_index]            
            
          #  base_model = GeneticTree()
            base_model.set_params(**parameter_setting)

            base_model.fit(enforce_numpy(X_train_cv), enforce_numpy(y_train_cv))

            base_model_pred = base_model.predict(enforce_numpy(X_valid_cv))

            if metric not in ['f1', 'roc_auc']:
                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_valid_cv), np.round(base_model_pred))
            else:
                if metric == 'f1':
                    score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_valid_cv), np.round(base_model_pred), average='weighted')
                elif metric == 'roc_auc':
                    try:
                        if int(np.max(y_train_cv)+1) > 2:                            
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(tf.keras.utils.to_categorical(y_valid_cv, num_classes=int(max(pd.concat([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']]))+1))), np.round(tf.keras.utils.to_categorical(base_model_pred, num_classes=int(max(pd.concat([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']]))+1))), multi_class='ovo')
                        else:
                            #score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(tf.keras.utils.to_categorical(y_valid_cv)), np.round(tf.keras.utils.to_categorical(base_model_pred)))
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_valid_cv), np.round(base_model_pred))
                            
                    except ValueError:
                        score_base_model_cv = 0.5           
            score_base_model_cv_list.append(score_base_model_cv)
            
        score_base_model = np.mean(score_base_model_cv_list)

        return (score_base_model, parameter_setting)      
    

    #parameter_grid = ParameterGrid(parameter_dict)
    parameter_grid = ParameterSampler(n_iter = config['computation']['search_iterations'],
                                       param_distributions = parameter_dict,
                                       random_state = config['computation']['random_seed'])

    print('Number of Trials GenTree: ' + str(len(parameter_grid)))

    parallel_hpo = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend='loky') #loky #sequential multiprocessing
    grid_search_results = parallel_hpo(delayed(evaluate_parameter_setting_gen)(parameter_setting, dataset_dict, config) for parameter_setting in parameter_grid)           

    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    best_params = grid_search_results_sorted[0][1]
    best_score = grid_search_results_sorted[0][0]

   # base_model = GeneticTree()
    base_model.set_params(**best_params)

    start = timeit.default_timer()
    base_model.fit(enforce_numpy(dataset_dict['X_train']), enforce_numpy(dataset_dict['y_train']))       
    end = timeit.default_timer()  
    runtime = end - start 

    display(best_params)  
    
    
    best_params_complete = get_params_gentree(base_model, config)
    
    
    hpo_path_by_dataset =  './evaluation_results/hpo/GeneticTree/' + identifier + '.csv'
    Path('./evaluation_results/hpo/GeneticTree/').mkdir(parents=True, exist_ok=True)    

        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['score', list(best_params_complete.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)          
    writer.writerow(flatten_list([best_score, list(best_params_complete.values())]))
    file_by_dataset.close()  
    
    return (runtime, base_model)
    
    
    
    


    
def hpo_sklearn(identifier, dataset_dict, parameter_dict, config, metric='f1', greater_better=True):
    
    def evaluate_parameter_setting_sklearn(parameter_setting, dataset_dict, config):   
        
        if config['gdt']['objective'] == 'classification':
            cv_generator = StratifiedKFold(n_splits=config['computation']['cv_num'], shuffle=True, random_state=config['computation']['random_seed'])
        else:
            cv_generator = KFold(n_splits=config['computation']['cv_num'], shuffle=True, random_state=config['computation']['random_seed'])        

        X_train_valid = pd.concat([dataset_dict['X_train'], dataset_dict['X_valid']])
        y_train_valid = pd.concat([dataset_dict['y_train'], dataset_dict['y_valid']])

        score_base_model_cv_list = []
        for train_index, valid_index in cv_generator.split(X_train_valid, y_train_valid):
            #print("TRAIN:", train_index, "TEST:", test_index)
            if isinstance(X_train_valid, pd.DataFrame) or isinstance(X_train_valid, pd.Series):
                X_train_cv, X_valid_cv = X_train_valid.iloc[train_index], X_train_valid.iloc[valid_index]
            else:
                X_train_cv, X_valid_cv = X_train_valid[train_index], X_train_valid[valid_index] 
                
            if isinstance(y_train_valid, pd.DataFrame) or isinstance(y_train_valid, pd.Series):
                y_train_cv, y_valid_cv = y_train_valid.iloc[train_index], y_train_valid.iloc[valid_index]
            else:
                y_train_cv, y_valid_cv = y_train_valid[train_index], y_train_valid[valid_index]  
                
            if config['gdt']['objective'] == 'classification':
                 sklearn_model = DecisionTreeClassifier
            else:
                 sklearn_model = DecisionTreeRegressor                
                
            base_model = sklearn_model()
            base_model.set_params(**parameter_setting)

            base_model.fit(enforce_numpy(X_train_cv), enforce_numpy(y_train_cv))

            base_model_pred = base_model.predict(enforce_numpy(X_valid_cv))
            if metric not in ['f1', 'roc_auc']:
                score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_valid_cv), np.round(base_model_pred))
            else:
                if metric == 'f1':
                    score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_valid_cv), np.round(base_model_pred), average='weighted')
                elif metric == 'roc_auc':
                    try:
                        if int(np.max(y_train_cv)+1) > 2:                            
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(tf.keras.utils.to_categorical(y_valid_cv, num_classes=int(max(pd.concat([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']]))+1))), np.round(tf.keras.utils.to_categorical(base_model_pred, num_classes=int(max(pd.concat([dataset_dict['y_train'], dataset_dict['y_valid'], dataset_dict['y_test']]))+1))), multi_class='ovo')
                        else:
                            #score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(tf.keras.utils.to_categorical(y_valid_cv)), np.round(tf.keras.utils.to_categorical(base_model_pred)))    
                            score_base_model_cv = sklearn.metrics.get_scorer(metric)._score_func(enforce_numpy(y_valid_cv), np.round(base_model_pred))       
                            
                    except ValueError:
                        score_base_model_cv = 0.5
                    
            score_base_model_cv_list.append(score_base_model_cv)
            
        score_base_model = np.mean(score_base_model_cv_list)

        return (score_base_model, parameter_setting)    
    
    

    #parameter_grid = ParameterGrid(parameter_dict)
    parameter_grid = ParameterSampler(n_iter = config['computation']['search_iterations'],
                                              param_distributions = parameter_dict,
                                              random_state = config['computation']['random_seed'])

    print('Number of Trials sklearn: ' + str(len(parameter_grid)))

    parallel_hpo = Parallel(n_jobs=config['computation']['n_jobs'], verbose=3, backend='loky') #loky #sequential multiprocessing
    grid_search_results = parallel_hpo(delayed(evaluate_parameter_setting_sklearn)(parameter_setting, dataset_dict, config) for parameter_setting in parameter_grid)           

    grid_search_results_sorted = sorted(grid_search_results, key=lambda tup: tup[0], reverse=greater_better) 

    
    best_params = grid_search_results_sorted[0][1]
    best_score = grid_search_results_sorted[0][0]

    if config['gdt']['objective'] == 'classification':
        sklearn_model = DecisionTreeClassifier
        model_identifier = 'Sklearn_class'
    else:
        sklearn_model = DecisionTreeRegressor      
        model_identifier = 'Sklearn_reg'
        
    
    base_model = sklearn_model()
    base_model.set_params(**best_params)

    start = timeit.default_timer()
    base_model.fit(dataset_dict['X_train'], dataset_dict['y_train'])       
    end = timeit.default_timer()  
    runtime = end - start

    display(best_params)        
    
    best_params_complete = base_model.get_params()
    
    #import csv
    
    hpo_path_by_dataset =  './evaluation_results/hpo/' + model_identifier + '/' + identifier + '.csv'
    Path('./evaluation_results/hpo/' + model_identifier + '/').mkdir(parents=True, exist_ok=True)    
        
    if not os.path.isfile(hpo_path_by_dataset):
        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)          
        writer.writerow(flatten_list(['score', list(best_params_complete.keys())]))

    file_by_dataset = open(hpo_path_by_dataset, 'a+')
    writer = csv.writer(file_by_dataset)          
    writer.writerow(flatten_list([best_score, list(best_params_complete.values())]))
    file_by_dataset.close()        
    
    return (runtime, base_model)




def structure_evaluation_results(evaluation_results, 
                                 benchmark_dict,
                                 identifier_list,
                                 config,
                                 metrics = [],
                                metric_identifer='test'):
    
    for i, result in enumerate(evaluation_results):
        if i == 0:
            model_dict = result[0]
            scores_dict = result[1]
            dataset_dict = result[2]
        else: 
            model_dict = mergeDict(model_dict, result[0])
            scores_dict = mergeDict(scores_dict, result[1])
            dataset_dict = mergeDict(dataset_dict, result[2])    

    model_identifier_list = flatten_list(['GDT', list(benchmark_dict.keys())])           
            
    metric_identifer = '_' + metric_identifer
    index = identifier_list
    columns = flatten_list([[[approach + ' ' + metric + '_mean', approach + ' ' + metric + '_max', approach + ' ' + metric + '_std'] for metric in metrics] for approach in model_identifier_list])

    
    results_list = []
    runtime_list = []
    for model_identifier in model_identifier_list:
        results = None
        
        for metric in metrics:
            scores = [scores_dict[identifier][model_identifier][metric + metric_identifer] for identifier in identifier_list]

            scores_mean = np.mean(scores, axis=1) if config['computation']['trials'] > 1 else scores

            scores_max = np.max(scores, axis=1) if config['computation']['trials'] > 1 else scores

            scores_std = np.std(scores, axis=1) if config['computation']['trials'] > 1 else np.array([0.0] * len(scores))

            results_by_metric = np.vstack([scores_mean, scores_max, scores_std])

            if results is None:
                results = results_by_metric        
            else:
                results = np.vstack([results, results_by_metric])             
                
        results_list.append(results)
        
        runtimes = np.array([scores_dict[identifier][model_identifier]['runtime'] for identifier in identifier_list])
        runtime_list.append(runtimes)          
            
    scores_dataframe = pd.DataFrame(data=np.vstack(results_list).T, index = index, columns = columns)     
 
    columns = flatten_list([[model_identifier + ' ' + measure for model_identifier in model_identifier_list] for measure in ['Mean', 'Std', 'Max']])

    
    if config['computation']['trials'] > 1:
        runtime_results = pd.DataFrame(data=np.vstack([[np.mean(runtimes, axis=1) for runtimes in runtime_list], 
                                                                          [np.std(runtimes, axis=1) for runtimes in runtime_list], 
                                                                          [np.max(runtimes, axis=1) for runtimes in runtime_list]]).T, index=identifier_list, columns=columns)
    
    
    
    
    
    
    else:
        runtime_results = pd.DataFrame(data=np.vstack([[runtimes for runtimes in runtime_list], 
                                                                          [np.array([0.0] * runtimes.shape[0]) for runtimes in runtime_list], 
                                                                          [runtimes for runtimes in runtime_list]]).T, index=identifier_list, columns=columns)
        
        

        
        
    index = [index_name.split(' ')[1] for index_name in scores_dataframe.mean()[:scores_dataframe.shape[1]//len(model_identifier_list)].index]
    mean_result_dataframe = np.round(pd.DataFrame(data=np.vstack(np.array_split(scores_dataframe.mean(), len(model_identifier_list))).T, index=index, columns=model_identifier_list), 3)
           
    mean_runtime_results = np.round(pd.DataFrame(data=np.vstack(np.array_split(runtime_results.mean(), 3))[[0,2,1]], index=['runtime_mean', 'runtime_max', 'runtime_std'], columns=model_identifier_list), 3)

    
    mean_result_dataframe = pd.concat([mean_result_dataframe, mean_runtime_results])
    
    
    
    scores_dataframe_mean = scores_dataframe.mean()
    scores_dataframe_mean.name = 'MEAN'

    count_list = []
    for column in scores_dataframe.columns:
        column_metric_identifier = column.split(' ')[1]
        count_series = scores_dataframe[column]>=get_columns_by_name(scores_dataframe, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    scores_dataframe_count = pd.Series(count_list, index = scores_dataframe.columns)
    scores_dataframe_count.name = 'COUNT'    
    
    scores_dataframe = scores_dataframe.append(scores_dataframe_mean)              
    scores_dataframe = scores_dataframe.append(scores_dataframe_count)              

    runtime_results_mean = runtime_results.mean()
    runtime_results_mean.name = 'MEAN'
    
    count_list = []
    for column in runtime_results.columns:
        column_metric_identifier = column.split(' ')[1]
        count_series = runtime_results[column]>=get_columns_by_name(runtime_results, column_metric_identifier).max(axis=1)
        #count_series.drop('MEAN', inplace=True)
        count = count_series.sum()
        count_list.append(count)
    runtime_results_count = pd.Series(count_list, index = runtime_results.columns)
    runtime_results_count.name = 'COUNT'    
    
    runtime_results = runtime_results.append(runtime_results_mean)          
    runtime_results = runtime_results.append(runtime_results_count)              
    return scores_dataframe, runtime_results, mean_result_dataframe
        

    
def get_hpo_best_params_by_dataset(timestr, dataset_name):
    
    for depth in range(1, 20):
        try:
            #filepath = './evaluation_results/depth' + str(config['gdt']['depth']) + '/' + timestr + '/'
            filepath = './evaluation_results/depth' + str(depth) + '/' + timestr + '/'
            if 'BIN' in dataset_name:
                with open(filepath + 'hpo_best_params_classification_binary.pickle', 'rb') as file:
                    hpo_best_params = pickle.load(file, protocol=pickle.HIGHEST_PROTOCOL)
            elif 'MULT' in dataset_name:
                with open(filepath + 'hpo_best_params_classification_multi.pickle', 'rb') as file:
                    hpo_best_params = pickle.load(file, protocol=pickle.HIGHEST_PROTOCOL)    
            elif 'REG' in dataset_name:
                with open(filepath + 'hpo_best_params_classification_regression.pickle', 'rb') as file:
                    hpo_best_params = pickle.load(file, protocol=pickle.HIGHEST_PROTOCOL)       
        except:
            pass
        
    return hpo_best_params[dataset_name] 


def write_hpo_results_to_csv(hpo_results_real_world, identifier_list):
        
    hpo_path = './evaluation_results/hpo_best/gdt/'
    Path(hpo_path).mkdir(parents=True, exist_ok=True)    

    for identifier in identifier_list:
        hpo_path_by_dataset = hpo_path + identifier + '.csv'
        #file_by_dataset = open(hpo_path_by_dataset, 'r+')

        best_params_dict = flatten_dict(hpo_results_real_world[identifier][0]['parameters_complete'])
        best_params_score =  hpo_results_real_world[identifier][0]['GDT mean (mean)']
        headers_by_dataset_dict = flatten_list(['score', list(best_params_dict.keys())])#list(best_params_dict.keys())

        if os.path.isfile(hpo_path_by_dataset):
            file_by_dataset = open(hpo_path_by_dataset, 'r+')
            reader = csv.reader(file_by_dataset)
            headers_by_dataset_file = next(reader, None)

            counter = 1
            while not headers_by_dataset_dict == headers_by_dataset_file:
                hpo_path_by_dataset = hpo_path + identifier + str(counter) + '.csv'

                if not os.path.isfile(hpo_path_by_dataset):
                    file_by_dataset = open(hpo_path_by_dataset, 'a+')
                    writer = csv.writer(file_by_dataset)
                    writer.writerow(headers_by_dataset_dict)
                    break

                file_by_dataset = open(hpo_path_by_dataset, 'r+')

                reader = csv.reader(file_by_dataset)
                headers_by_dataset_file = next(reader, None)  
                counter += 1
        else:
            file_by_dataset = open(hpo_path_by_dataset, 'a+')
            writer = csv.writer(file_by_dataset)   
            writer.writerow(headers_by_dataset_dict)

        file_by_dataset = open(hpo_path_by_dataset, 'a+')
        writer = csv.writer(file_by_dataset)
        writer.writerow(flatten_list([best_params_score, list(best_params_dict.values())]))
        file_by_dataset.close()    
    
    hpo_path = './evaluation_results/hpo_complete/gdt/'
    Path(hpo_path).mkdir(parents=True, exist_ok=True)    

    for identifier in identifier_list:
        hpo_path_by_dataset = hpo_path + identifier + '.csv'
        #file_by_dataset = open(hpo_path_by_dataset, 'r+')

        for setting_number in range(len(hpo_results_real_world[identifier])):
        
            best_params_dict = flatten_dict(hpo_results_real_world[identifier][setting_number]['parameters_complete'])
            best_params_score =  hpo_results_real_world[identifier][setting_number]['GDT mean (mean)']
            headers_by_dataset_dict = flatten_list(['score', list(best_params_dict.keys())])#list(best_params_dict.keys())

            if os.path.isfile(hpo_path_by_dataset):
                file_by_dataset = open(hpo_path_by_dataset, 'r+')
                reader = csv.reader(file_by_dataset)
                headers_by_dataset_file = next(reader, None)

                counter = 1
                while not headers_by_dataset_dict == headers_by_dataset_file:
                    hpo_path_by_dataset = hpo_path + identifier + str(counter) + '.csv'

                    if not os.path.isfile(hpo_path_by_dataset):
                        file_by_dataset = open(hpo_path_by_dataset, 'a+')
                        writer = csv.writer(file_by_dataset)
                        writer.writerow(headers_by_dataset_dict)
                        break

                    file_by_dataset = open(hpo_path_by_dataset, 'r+')

                    reader = csv.reader(file_by_dataset)
                    headers_by_dataset_file = next(reader, None)  
                    counter += 1
            else:
                file_by_dataset = open(hpo_path_by_dataset, 'a+')
                writer = csv.writer(file_by_dataset)   
                writer.writerow(headers_by_dataset_dict)

            file_by_dataset = open(hpo_path_by_dataset, 'a+')
            writer = csv.writer(file_by_dataset)
            writer.writerow(flatten_list([best_params_score, list(best_params_dict.values())]))
            file_by_dataset.close()

def read_best_hpo_result_from_csv_benchmark(dataset_name,
                                            model_identifier, 
                                            return_best_only=True, 
                                            ascending=False):
    hpo_path_by_dataset = './evaluation_results/hpo/' + model_identifier + '/' + dataset_name  +  '.csv'    
    
    hpo_results = pd.read_csv(hpo_path_by_dataset) 
    #print(hpo_results.shape)
    #print(hpo_results.columns)
    #print(hpo_results.values[0])
    hpo_results.sort_values(by=['score'], ascending=ascending, inplace=True)    
    
    hpo_results_best = hpo_results.iloc[0]
    
    hpo_results_best_keys = list(hpo_results_best.index)
    hpo_results_best_values = list(hpo_results_best.values)

    hpo_results_best_dict = {'model': {}}
    for key, value in zip(hpo_results_best_keys, hpo_results_best_values):
        if key == 'score':
            hpo_results_best_dict[key] = value
        else:
            if pd.isna(value):
                hpo_results_best_dict['model'][key] = None
            else:
                hpo_results_best_dict['model'][key] = value

    if return_best_only:
        return hpo_results_best_dict
    
    return hpo_results_best_dict, hpo_results    
    
    
def read_best_hpo_result_from_csv(dataset_name, 
                                  counter='', 
                                  return_best_only=True, 
                                  ascending=False):
    hpo_path_by_dataset = './evaluation_results/hpo_best/gdt/' + dataset_name + str(counter) +  '.csv'    
    
    hpo_results = pd.read_csv(hpo_path_by_dataset) 
    hpo_results.sort_values(by=['score'], ascending=ascending, inplace=True)
    
    hpo_results_best = hpo_results.iloc[0]
    
    hpo_results_best_keys = list(hpo_results_best.index)
    hpo_results_best_values = list(hpo_results_best.values)

    hpo_results_best_keys_nested = []
    for key in hpo_results_best_keys:
        key_nested = key.split('__')
        hpo_results_best_keys_nested.append(key_nested)

    hpo_results_best_dict = {}
    for key_list, value in zip(hpo_results_best_keys_nested, hpo_results_best_values):
        current_dict = hpo_results_best_dict
        for i in range(len(key_list)):
            current_key = key_list[i]
            if i < len(key_list)-1:
                if current_key not in hpo_results_best_dict.keys():
                    current_dict[current_key] = {}

                current_dict = current_dict[current_key]
            else:
                if current_key not in hpo_results_best_dict.keys():
                    if pd.isna(value):
                        current_dict[current_key] = None
                    else:
                        current_dict[current_key] = value                    
                 

    if return_best_only:
        return hpo_results_best_dict
    
    
    return hpo_results_best_dict, hpo_results



def write_latex_table_top(f):
    f.write('\\begin{table}[htb]' + '\n')
    f.write('\\centering' + '\n')
    f.write('\\resizebox{\columnwidth}{!}{' + '\n')
    f.write('%\\begin{threeparttable}' + '\n')

    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + '\n')
    
    
def write_latex_table_bottom(f, model_type):
    f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + '\n')

    f.write('%\\begin{tablenotes}' + '\n')
    f.write('%\\item[a] \\footnotesize' + '\n')
    f.write('%\\item[b] \\footnotesize' + '\n')
    f.write('%\\end{tablenotes}' + '\n')
    f.write('%\\end{threeparttable}' + '\n')
    f.write('}' + '\n')
    f.write('\\caption{\\textbf{' + model_type +' Performance Comparison.} We report the train and test f1-score (mean $\pm$ stdev over 10 trials) and dataset specification. We also report the ranking of each approach for the corresponding dataset in brackets.}' + '\n')
    f.write('\\label{tab:eval-results_' + model_type.split(' ')[0] + '}' + '\n')
    f.write('\\end{table}' + '\n')
    
def add_hline(latex: str, index: int) -> str:
    """
    Adds a horizontal `index` lines before the last line of the table

    Args:
        latex: latex table
        index: index of horizontal line insertion (in lines)
    """
    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines)#.replace('NaN', '')



def combine_mean_std_columns(dataframe, reverse=False, highlight=True):
    dataframe_combined_list = [] 
    column_name_list = []

    if reverse:
        mean_columns_comparator = np.argsort(np.argsort(dataframe.iloc[:,::2].values)) + 1
    else:
        mean_columns_comparator = np.argsort(np.argsort(-dataframe.iloc[:,::2].values)) + 1
    #dataframe.iloc[:,::2]

    index_list = []
    
    for index_name in dataframe.index[:-2]:
        index_list.append(index_name.split(':')[1])
    
    for column_index in range(len(dataframe.columns)//2):
        column_index_mean = 2*column_index
        column_index_std = 2*column_index+1

        column_mean = dataframe.iloc[:,column_index_mean]
        column_std = dataframe.iloc[:,column_index_std]

        column_name = column_mean.name.split(' ')[0]
        column_name_list.append(column_name)

        #combined_column = '$' + np.round(column_mean, 3).astype(str) + ' \pm ' +  np.round(column_std, 3).astype(str) + ' (' +  mean_columns_comparator[:,column_index].astype(str) + ')$'

        combined_column_list = []
        if mean_columns_comparator[:,0].shape[0] // 3 == 1:
            rank_list = mean_columns_comparator[:,column_index]
        else:
            rank_list_raw = mean_columns_comparator[:,column_index]
            rank_list = []
            for i in range(mean_columns_comparator[:,0].shape[0] // 3):
                if i == mean_columns_comparator[:,0].shape[0] // 3 - 1:
                    rank_list.append(np.argsort(np.argsort(-1 * rank_list_raw[i*3:(i+1)*3]))+1)     
                else:
                    rank_list.append(np.argsort(np.argsort(rank_list_raw[i*3:(i+1)*3]))+1)     

            rank_list = flatten_list(rank_list)        
        
        for value_mean, value_std, rank in zip(column_mean, column_std, mean_columns_comparator[:,column_index]):
            if highlight:
                if rank == 1:

                    value = '$\mathbf{' + '{:.3f}'.format(round(value_mean, 3))+ ' \pm ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')}$'
                else:
                    value = '$' + '{:.3f}'.format(round(value_mean, 3)) + ' \pm ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')$'
            else:
                value = '{:.3f}'.format(round(value_mean, 3)) + ' ± ' + '{:.3f}'.format(round(value_std, 3)) + ' (' + rank.astype(str) + ')'
            combined_column_list.append(value)
        dataframe_combined_list.append(combined_column_list)

    dataframe_combined = pd.DataFrame(data=np.array(dataframe_combined_list).T[:-2,:], columns=column_name_list, index=index_list)

    result_column_list = []
    
    
    if dataframe_combined.shape[1] // 3 == 1:
        rank_list = (np.argsort(np.argsort(-dataframe.loc['MEAN'][::2].values)) + 1)
    else:
        rank_list_raw = (np.argsort(np.argsort(-dataframe.loc['MEAN'][::2].values)) + 1)
        rank_list = []
        for i in range(dataframe_combined.shape[1] // 3):
            if i == dataframe_combined.shape[1] // 3 - 1:
                rank_list.append(np.argsort(np.argsort(-1 * rank_list_raw[i*3:(i+1)*3]))+1)     
            else:
                rank_list.append(np.argsort(np.argsort(rank_list_raw[i*3:(i+1)*3]))+1)     
                
        rank_list = flatten_list(rank_list)
        
    for result_value, result_rank in zip(dataframe.loc['MEAN'][::2].values, rank_list):
        if highlight:
            if result_rank == 1:
                result_column = '$\mathbf{' + '{:0.3f}'.format(round(result_value, 3)) + ' (' + result_rank.astype(str) + ')}$'
            else:
                result_column = '${:0.3f}'.format(round(result_value, 3)) + ' (' + result_rank.astype(str) + ')$'   
        else:
            result_column = '{:0.3f}'.format(round(result_value, 3)) + ' (' + result_rank.astype(str) + ')'   
        result_column_list.append(result_column)

    #result_column = '$' + np.array(['{:0.3f}'.format(round(x, 3)) for x in dataframe.loc['MEAN'][::2].values]).astype(object) + ' (' + (np.argsort(np.argsort(-dataframe.loc['MEAN'][::2].values)) + 1).astype(str) + ')$'
    result_column_pandas = pd.DataFrame(data=np.array([result_column_list]), columns=column_name_list, index=['Mean'])
        
    dataframe_combined = dataframe_combined.append(result_column_pandas)
    
    return dataframe_combined    
    
def plot_table_save_results(benchmark_dict,
                            evaluation_results_real_world,
                            identifier_list,
                            #scores_dataframe_real_world,
                            #runtime_results,
                            #mean_result_dataframe_real_world,
                            #metrics,
                            identifier_string,     
                            filepath,               
                            config,
                            plot_runtime=False,
                            terminal_output=False):
    if (identifier_string.split('_')[0] == 'regression' and isinstance(config['computation']['eval_metric_reg'], list)) or (not identifier_string.split('_')[0] == 'regression' and isinstance(config['computation']['eval_metric_class'], list)):
        
        if identifier_string.split('_')[0] == 'regression':
            metrics = config['computation']['metrics_reg']
            select_metric_name_list = config['computation']['eval_metric_reg']
        else:
            metrics = config['computation']['metrics_class']
            select_metric_name_list = config['computation']['eval_metric_class']
    
        for select_metric_name in select_metric_name_list:  

            (scores_dataframe_real_world, 
             runtime_results, 
             mean_result_dataframe_real_world) = structure_evaluation_results(evaluation_results = evaluation_results_real_world,
                                                                                     benchmark_dict = benchmark_dict,
                                                                                     identifier_list = identifier_list,
                                                                                     config = config,
                                                                                     metrics = metrics,
                                                                                     metric_identifer=identifier_string.split('_')[1])    

            model_identifier_list = flatten_list(['GDT', list(benchmark_dict.keys())])           
            runtime_results.columns = [column + str(' Runtime') for column in runtime_results.columns]    

            scores_dataframe_real_world.to_csv(filepath + 'scores_dataframe_' + select_metric_name + '_' + identifier_string)
            runtime_results.to_csv(filepath + 'runtime_results_' + select_metric_name + '_' + identifier_string)
            mean_result_dataframe_real_world.to_csv(filepath + 'mean_result_dataframe_' + select_metric_name + '_' + identifier_string)       

            reorder = flatten_list([[(i*len(metrics))+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
            scores_dataframe_real_world_combined = pd.concat([scores_dataframe_real_world[scores_dataframe_real_world.columns[0::3]].iloc[:,reorder], runtime_results.iloc[:,:len(model_identifier_list)]], axis=1)
            smaller_better_names = ['neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_error_mean', 'neg_mean_squared_error_mean', 'Runtime']
            smaller_better_name_by_index = [any(smaller_better_name in column for smaller_better_name in smaller_better_names) for column in scores_dataframe_real_world_combined.columns]
            highlight_index = list(scores_dataframe_real_world_combined.index)
            highlight_index.remove('MEAN')
            highlight_index.remove('COUNT')



            columnnames_select_metric = []
            for column in scores_dataframe_real_world.columns:
                if select_metric_name in column and 'max' not in column:
                    columnnames_select_metric.append(column)
    
            if not terminal_output:
                display(scores_dataframe_real_world[columnnames_select_metric].style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                                               else 'background-color: #FF6961' if (v == min(x[::2]))
                                                                                               else 'background-color: #9CC95C' if (v == max(x[::2]))
                                                                                               else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))

                
            timestr = filepath.split('/')[-2]

            with open("./evaluation_results/latex_tables/" + timestr + "/latex_table_" + select_metric_name + '_' + identifier_string + ".tex", "a+") as f:
                write_latex_table_top(f)
                f.write(add_hline(combine_mean_std_columns(scores_dataframe_real_world[columnnames_select_metric]).to_latex(index=True, bold_rows=False, escape=False), 1))
                write_latex_table_bottom(f, identifier_string)
                f.write('\n\n')        

            
    
    else:
    
        if identifier_string.split('_')[0] == 'regression':
            metrics = config['computation']['metrics_reg']
            select_metric_name = config['computation']['eval_metric_reg']
        else:
            metrics = config['computation']['metrics_class']
            select_metric_name = config['computation']['eval_metric_class']


        (scores_dataframe_real_world, 
         runtime_results, 
         mean_result_dataframe_real_world) = structure_evaluation_results(evaluation_results = evaluation_results_real_world,
                                                                                 benchmark_dict = benchmark_dict,
                                                                                 identifier_list = identifier_list,
                                                                                 config = config,
                                                                                 metrics = metrics,
                                                                                 metric_identifer=identifier_string.split('_')[1])    

        model_identifier_list = flatten_list(['GDT', list(benchmark_dict.keys())])           
        runtime_results.columns = [column + str(' Runtime') for column in runtime_results.columns]    

        scores_dataframe_real_world.to_csv(filepath + 'scores_dataframe_' + identifier_string)
        runtime_results.to_csv(filepath + 'runtime_results_' + identifier_string)
        mean_result_dataframe_real_world.to_csv(filepath + 'mean_result_dataframe_' + identifier_string)       

        reorder = flatten_list([[(i*len(metrics))+j for i in range(len(model_identifier_list))] for j in range(len(metrics))])
        scores_dataframe_real_world_combined = pd.concat([scores_dataframe_real_world[scores_dataframe_real_world.columns[0::3]].iloc[:,reorder], runtime_results.iloc[:,:len(model_identifier_list)]], axis=1)
        smaller_better_names = ['neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_percentage_error_mean', 'neg_mean_absolute_error_mean', 'neg_mean_squared_error_mean', 'Runtime']
        smaller_better_name_by_index = [any(smaller_better_name in column for smaller_better_name in smaller_better_names) for column in scores_dataframe_real_world_combined.columns]
        highlight_index = list(scores_dataframe_real_world_combined.index)
        highlight_index.remove('MEAN')
        highlight_index.remove('COUNT')



        columnnames_select_metric = []
        for column in scores_dataframe_real_world.columns:
            if select_metric_name in column and 'max' not in column:
                columnnames_select_metric.append(column)

        if not terminal_output:
            display(scores_dataframe_real_world[columnnames_select_metric].style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                                           else 'background-color: #FF6961' if (v == min(x[::2]))
                                                                                           else 'background-color: #9CC95C' if (v == max(x[::2]))
                                                                                           else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))


        timestr = filepath.split('/')[-2]

        with open("./evaluation_results/latex_tables/" + timestr + "/latex_table_" + identifier_string + ".tex", "a+") as f:
            write_latex_table_top(f)
            f.write(add_hline(combine_mean_std_columns(scores_dataframe_real_world[columnnames_select_metric]).to_latex(index=True, bold_rows=False, escape=False), 1))
            write_latex_table_bottom(f, identifier_string)
            f.write('\n\n')        
 

    if plot_runtime:
        reorder = flatten_list([[(i*3)+j for i in range(2)] for j in range(3)])
        runtime_results_latex = runtime_results.iloc[:,:-(len(benchmark_dict)+1)].iloc[:,reorder]
        if not terminal_output:
            display(runtime_results_latex.style.apply(lambda x: ['background-color: #779ECB' if (v == np.mean(x[::2]))
                                                                   else 'background-color: #FF6961' if (v == max(x[::2]))
                                                                   else 'background-color: #9CC95C' if (v == min(x[::2]))
                                                                   else '' for i, v in enumerate(x)], axis = 1, subset=pd.IndexSlice[highlight_index, :]))
        else:
            terminal_combined_df = combine_mean_std_columns(pd.concat([scores_dataframe_real_world[columnnames_select_metric], runtime_results_latex], axis=1, join="inner"), highlight=False)

            columnnames = terminal_combined_df.columns

            columnnames_new = []
            for i, columnname in enumerate(columnnames):
                if i >= len(columnnames)//2:
                    columnnames_new.append(columnname + ' Runtime')
                else:
                    columnnames_new.append(columnname)

            print(tabulate(terminal_combined_df, headers='keys', tablefmt='psql'))        


        with open("./evaluation_results/latex_tables/" + timestr + "/latex_table_runtime_" + identifier_string + ".tex", "w+") as f:
            write_latex_table_top(f)
            f.write(add_hline(combine_mean_std_columns(runtime_results_latex, reverse=True).to_latex(index=True, bold_rows=False, escape=False), 1))
            write_latex_table_bottom(f, 'RUNTIME')
            f.write('\n\n')          
    
    


def plot_dt_comparison(evaluation_results_real_world,
                      identifier_list,
                      identifier_string,
                      timestr,
                      config):
    
    plot_identifier = identifier_list[0]     
    
    model_dict = [evaluation_result_real_world[0] for evaluation_result_real_world in evaluation_results_real_world]
    scores_dict = [evaluation_result_real_world[1] for evaluation_result_real_world in evaluation_results_real_world]        
    dataset_dict = [evaluation_result_real_world[2] for evaluation_result_real_world in evaluation_results_real_world]         
    
    if True:
        best_identifier = None
        best_plot_index = None
        difference_best = -np.inf

        for identifier in identifier_list:
            for plot_index in range(config['computation']['trials']):
                difference_current = scores_dict[plot_index][identifier]['GDT']['f1_test'] - scores_dict[plot_index][identifier]['sklearn']['f1_test']
                if difference_current > difference_best:
                    difference_best = difference_current
                    best_identifier = identifier
                    best_plot_index = plot_index  

        print(best_identifier)
        print(best_plot_index)
        plot_index = best_plot_index#0
        identifier = best_identifier#identifier_list[0]
    elif True:
        best_index = 0
        
        differences_dict = {}
        for identifier in identifier_list:
            differences_list = []
            
            for plot_index in range(config['computation']['trials']):
                
                difference_current = scores_dict[plot_index][identifier]['GDT']['f1_test'] - scores_dict[plot_index][identifier]['sklearn']['f1_test']
                differences_list.append(-difference_current)

            differences_dict[identifier] = np.argsort(differences_list)

        identifier = plot_identifier
        plot_index = differences_dict[identifier][best_index]
        
        print(identifier)
        print(plot_index)            
        
    else:
        plot_index = 0
        identifier = plot_identifier
    print('F1 Score (GDT):\t\t', scores_dict[plot_index][identifier]['GDT']['f1_test'])
    print('Accuracy Score (GDT):\t', scores_dict[plot_index][identifier]['GDT']['accuracy_test'])
    print('ROC AUC Score (GDT):\t', scores_dict[plot_index][identifier]['GDT']['roc_auc_test'])
    plt.figure(figsize=(15,8))
    image = model_dict[plot_index][identifier]['GDT'].plot(normalizer_list=dataset_dict[plot_index][identifier]['normalizer_list'])
    display(image)
    
    print('F1 Score (CART):\t', scores_dict[plot_index][identifier]['sklearn']['f1_test'])
    print('Accuracy Score (CART):\t', scores_dict[plot_index][identifier]['sklearn']['accuracy_test'])
    print('ROC AUC Score (CART):\t', scores_dict[plot_index][identifier]['sklearn']['roc_auc_test'])
    plt.figure(figsize=(15,8))
    plot_tree(model_dict[plot_index][identifier]['sklearn'], fontsize=10) 
    plt.show()
        
    os.makedirs(os.path.dirname("./evaluation_results/plots/" + timestr +"/"), exist_ok=True)
        
    filehandler = open('./evaluation_results/plots/' + timestr + '/GDT_' + identifier + '_' + identifier_string + '.pickle', "wb")
    dill.dump(model_dict[plot_index][identifier]['GDT'], filehandler)
    filehandler.close()

    file = open('./evaluation_results/plots/' + timestr + '/GDT_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    GDT_loaded = dill.load(file)
    file.close()

    filehandler = open('./evaluation_results/plots/' + timestr + '/sklearn_' + identifier + '_' + identifier_string  + '.pickle', "wb")
    dill.dump(model_dict[plot_index][identifier]['sklearn'], filehandler)
    filehandler.close()

    file = open('./evaluation_results/plots/' + timestr + '/sklearn_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    sklearn_loaded = dill.load(file)
    file.close()    
    
    filehandler = open('./evaluation_results/plots/' + timestr + '/normalizer_list_' + identifier + '_' + identifier_string  + '.pickle', "wb")
    dill.dump(dataset_dict[plot_index][identifier]['normalizer_list'], filehandler)
    filehandler.close()

    file = open('./evaluation_results/plots/' + timestr + '/normalizer_list_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    normalizer_list_loaded = dill.load(file)
    file.close()    
    
    filehandler = open('./evaluation_results/plots/' + timestr + '/data_dict_' + identifier + '_' + identifier_string  + '.pickle', "wb")
    dill.dump(dataset_dict[plot_index][identifier], filehandler)
    filehandler.close()

    file = open('./evaluation_results/plots/' + timestr + '/data_dict_' + identifier + '_' + identifier_string  + '.pickle', 'rb')
    data_dict_list_loaded = dill.load(file)
    file.close()    
    
    
    
def get_benchmark_dict(config, eval_identifier):
    benchmark_dict = {}

    for key, value in config['benchmarks'].items():
        if value == True:
            if key == 'GeneticTree' and eval_identifier == 'regression':
                pass
            else:
                benchmark_dict[key] = None         
    
    return benchmark_dict

def prepare_training_for_streams(identifier, config: dict):
    
    #tf.random.set_seed(config['computation']['random_seed'])
    #np.random.seed(config['computation']['random_seed'])
    #random.seed(config['computation']['random_seed'])  
    
    config_test = deepcopy(config)
    #config_test['gdt']['epochs'] = 100
    if 'REG' not in identifier:
        metrics = ['f1', 'roc_auc', 'accuracy']
        sklearn_model = DecisionTreeClassifier
        
        if 'BIN' in identifier:
            config_test['gdt']['objective'] = 'classification'
            if 'loss' not in config_test['gdt']:
                config_test['gdt']['loss'] = 'crossentropy'     
            config_test['gdt']['normalize'] = None

        elif 'MULT' in identifier:
            config_test['gdt']['objective'] = 'classification'    
            if 'loss' not in config_test['gdt']:
                config_test['gdt']['loss'] = 'kl_divergence'
            config_test['gdt']['normalize'] = None           
        
    else:
        metrics = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        sklearn_model = DecisionTreeRegressor    
        
        config_test['gdt']['objective'] = 'regression'
        if 'loss' not in config_test['gdt']:
            config_test['gdt']['loss'] = 'mse'         
        if 'normalize' not in config_test['gdt']:
            config_test['gdt']['normalize'] = 'mean'     
            
    return config_test, metrics

    
    
    

def prepare_training(identifier, config: dict):
    
    tf.random.set_seed(config['computation']['random_seed'])
    np.random.seed(config['computation']['random_seed'])
    random.seed(config['computation']['random_seed'])  
    
    config_test = deepcopy(config)
    #config_test['gdt']['epochs'] = 100
    if 'REG' not in identifier:
        metrics = ['f1', 'roc_auc', 'accuracy']
        sklearn_model = DecisionTreeClassifier
        
        if 'BIN' in identifier:
            config_test['gdt']['objective'] = 'classification'
            if 'loss' not in config_test['gdt']:
                config_test['gdt']['loss'] = 'crossentropy'     
            config_test['gdt']['normalize'] = None

        elif 'MULT' in identifier:
            config_test['gdt']['objective'] = 'classification'    
            if 'loss' not in config_test['gdt']:
                config_test['gdt']['loss'] = 'kl_divergence'
            config_test['gdt']['normalize'] = None           
        
    else:
        metrics = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        sklearn_model = DecisionTreeRegressor    
        
        config_test['gdt']['objective'] = 'regression'
        if 'loss' not in config_test['gdt']:
            config_test['gdt']['loss'] = 'mse'         
        if 'normalize' not in config_test['gdt']:
            config_test['gdt']['normalize'] = 'mean'     
            
    dataset_dict = {}
    
    dataset_dict = get_preprocessed_dataset(identifier,
                                            random_seed=config_test['computation']['random_seed'],
                                            config=config_test,
                                            verbosity=0)    
    
    
    number_of_classes = len(np.unique(np.concatenate([dataset_dict['y_train'].values, dataset_dict['y_valid'].values, dataset_dict['y_test'].values])))

    
    dataset_dict['number_of_classes'] = number_of_classes
    
    dataset_dict['number_of_variables'] = dataset_dict['X_train'].shape[1]
    
    return dataset_dict, config_test, metrics
    
def prepare_score_dict(config):
    scores_dict = {'GDT': {}}

    for key, value in config['benchmarks'].items():
        if value == True:
            if config['gdt']['objective'] == 'regression' and key == 'GeneticTree':
                pass
            else:
                scores_dict[key] = {}    
                
    return scores_dict

def save_scores(dataset_name, scores_GDT, scores_VFDT, scores_CVFDT, VFDT_classifier, CVFDT_classifier, config, timer):
    #output performance
#Global:(Average, Median)  Sliding:() 
    results = pd.DataFrame(columns=['Metric','Operation','GDT','VFDT','CVFDT'])
    results.loc[len(results)] =    ['f1','median',round(np.median(scores_GDT['f1']),4),round(np.median(scores_VFDT['f1']),4),round(np.median(scores_CVFDT['f1']),4)]
    results.loc[len(results)] = ['f1','average',round(np.average(scores_GDT['f1']),4),round(np.average(scores_VFDT['f1']),4),round(np.average(scores_CVFDT['f1']),4)]
    results.loc[len(results)] = ['f1','std',round(np.std(scores_GDT['f1']),4),round(np.std(scores_VFDT['f1']),4),round(np.std(scores_CVFDT['f1']),4)]
    results.loc[len(results)] = ['f1','min',round(np.min(scores_GDT['f1']),4),round(np.min(scores_VFDT['f1']),4),round(np.min(scores_CVFDT['f1']),4)]
    results.loc[len(results)] = ['f1','max',round(np.max(scores_GDT['f1']),4),round(np.max(scores_VFDT['f1']),4),round(np.max(scores_CVFDT['f1']),4)]
    results.loc[len(results)] = ['-','-','-','-','-']

    results.loc[len(results)] = ['accuracy','median',round(np.median(scores_GDT['acc']),4),round(np.median(scores_VFDT['acc']),4),round(np.median(scores_CVFDT['acc']),4)]
    results.loc[len(results)] = ['accuracy','average',round(np.average(scores_GDT['acc']),4),round(np.average(scores_VFDT['acc']),4),round(np.average(scores_CVFDT['acc']),4)]
    results.loc[len(results)] = ['accuracy','std',round(np.std(scores_GDT['acc']),4),round(np.std(scores_VFDT['acc']),4),round(np.std(scores_CVFDT['acc']),4)]
    results.loc[len(results)] = ['accuracy','min',round(np.min(scores_GDT['acc']),4),round(np.min(scores_VFDT['acc']),4),round(np.min(scores_CVFDT['acc']),4)]
    results.loc[len(results)] = ['accuracy','max',round(np.max(scores_GDT['acc']),4),round(np.max(scores_VFDT['acc']),4),round(np.max(scores_CVFDT['acc']),4)]
    results.loc[len(results)] = ['-','-','-','-','-']
    
    
    results.loc[len(results)] = ['kappa','median',round(np.median(scores_GDT['kappa']),4),round(np.median(scores_VFDT['kappa']),4),round(np.median(scores_CVFDT['kappa']),4)]
    results.loc[len(results)] = ['kappa','average',round(np.average(scores_GDT['kappa']),4),round(np.average(scores_VFDT['kappa']),4),round(np.average(scores_CVFDT['kappa']),4)]
    results.loc[len(results)] = ['kappa','std',round(np.std(scores_GDT['kappa']),4),round(np.std(scores_VFDT['kappa']),4),round(np.std(scores_CVFDT['kappa']),4)]
    results.loc[len(results)] = ['kappa','min',round(np.min(scores_GDT['kappa']),4),round(np.min(scores_VFDT['kappa']),4),round(np.min(scores_CVFDT['kappa']),4)]
    results.loc[len(results)] = ['kappa','max',round(np.max(scores_GDT['kappa']),4),round(np.max(scores_VFDT['kappa']),4),round(np.max(scores_CVFDT['kappa']),4)]
    results.loc[len(results)] = ['-','-','-','-','-']
    

    results.loc[len(results)] = ['training time','total',round(timer['training']['GDT'],4),round(timer['training']['VFDT'],4),round(timer['training']['CVFDT'],4)]
    results.loc[len(results)] = ['prediction time','total',round(timer['prediction']['GDT'],4),round(timer['prediction']['VFDT'],4),round(timer['prediction']['CVFDT'],4)]

    results.to_excel('results/'+dataset_name+'/scores_'+dataset_name+'.xlsx')
    
                     
    #VFDT_Hyperparamas                 
    VFDT_Hyperparamas = str(VFDT_classifier.get_info())
    f = open('results/'+dataset_name+'/VFDT_Hyperparamas'+dataset_name+".txt", "w")
    f.write(VFDT_Hyperparamas)
    f.close()
      
    #CVFDT_Hyperparamas                 
    CVFDT_Hyperparamas = str(CVFDT_classifier.get_info())
    f = open('results/'+dataset_name+'/CVFDT_Hyperparamas'+dataset_name+".txt", "w")
    f.write(CVFDT_Hyperparamas)
    f.close()
                     
    #Parameters GDT
    #Dataset Name, depth, 'lr index', 'lr values', 'lr leaf', 'optimizer', 'batch size', 'pretrain_size', 'pretrain_epochs'
    GDT_Hyperparameters = pd.DataFrame(columns=['Dataset Name','depth','lr index', 'lr values', 'lr leaf', 'optimizer', 'batch size', 'pretrain_size', 'pretrain_epochs'])
    GDT_Hyperparameters.loc[len(GDT_Hyperparameters)] = [dataset_name, config['gdt']['depth'],config['gdt']['learning_rate_index'],config['gdt']['learning_rate_values'] ,config['gdt']['learning_rate_leaf'],config['gdt']['optimizer'], config['gdt']['batch_size'], config['computation']['pretrain_size'], config['gdt']['pretrain_epochs']]
    GDT_Hyperparameters.to_csv('results/'+dataset_name+'/GDT_Hyperparamas'+dataset_name+".csv")
    
    #Pairplot
    X = np.arange(0, len(scores_GDT['acc']), 1)
    with plt.style.context('default'):
 
        # Assign variables to the y axis part of the curve
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

        fig.tight_layout()

        # Plotting both the curves simultaneously
        axes[0].plot(X, scores_GDT['acc'],  label='GDT Acc', linewidth=0.4)
        axes[0].plot(X, scores_VFDT['acc'], label='VFDT Acc', linewidth=0.4)
        axes[0].plot(X, scores_CVFDT['acc'],  label='CVFDT Acc', linewidth=0.4)

        axes[1].plot(X, scores_GDT['f1'],  label='GDT_f1', linewidth=0.4)
        axes[1].plot(X, scores_VFDT['f1'], label='VFDT_f1', linewidth=0.4)
        axes[1].plot(X, scores_CVFDT['f1'], label='CVFDT_f1', linewidth=0.4)

        axes[2].plot(X, scores_GDT['kappa'], label='GDT_f1', linewidth=0.4)
        axes[2].plot(X, scores_VFDT['kappa'], label='VFDT_f1', linewidth=0.4)
        axes[2].plot(X, scores_CVFDT['kappa'], label='CVFDT_f1', linewidth=0.4)

        # Naming the x-axis, y-axis and the whole graph
        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("F1-score")
        axes[2].set_ylabel("Kappa")
        axes[1].set_title("Dataset: " + dataset_name)
        plt.legend()
    plt.tight_layout()
    plt.savefig('results/'+dataset_name+'/lineplots_'+dataset_name+'.png', dpi = 300.0)
    

    with plt.style.context('default'):
        #fig, ax = plt.subplots(figsize=(4, 3))
        plt.subplots(figsize=(4, 3))
        
        plt.plot(X, scores_GDT['f1'],  label='GDT_f1', linewidth=0.4) #465, 630
        plt.plot(X, scores_VFDT['f1'], label='VFDT_f1', linewidth=0.4)
        plt.plot(X, scores_CVFDT['f1'], label='CVFDT_f1', linewidth=0.4)

        # Naming the x-axis, y-axis and the whole graph
        plt.ylabel("F1-score")
        plt.title("Dataset: " + dataset_name)
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/'+dataset_name+'/f1_plot_'+dataset_name+'.png', dpi = 300.0)
    
     
    with plt.style.context('default'):
        plt.subplots(figsize=(4, 3))
        
        plt.plot(X, scores_GDT['acc'],  label='GDT_f1', linewidth=0.4) #465, 630
        plt.plot(X, scores_VFDT['acc'], label='VFDT_f1', linewidth=0.4)
        plt.plot(X, scores_CVFDT['acc'], label='CVFDT_f1', linewidth=0.4)

        # Naming the x-axis, y-axis and the whole graph
        plt.ylabel("Accuracy")
        plt.title("Dataset: " + dataset_name)
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/'+dataset_name+'/acc_plot_'+dataset_name+'.png', dpi = 300.0)
        
    plt.clf()  
    with plt.style.context('default'):
        plt.subplots(figsize=(4, 3))
        
        plt.plot(X, scores_GDT['kappa'],  label='GDT_f1', linewidth=0.4) #465, 630
        plt.plot(X, scores_VFDT['kappa'], label='VFDT_f1', linewidth=0.4)
        plt.plot(X, scores_CVFDT['kappa'], label='CVFDT_f1', linewidth=0.4)

        # Naming the x-axis, y-axis and the whole graph
        plt.ylabel("Kappa")
        plt.title("Dataset: " + dataset_name)
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/'+dataset_name+'/kappa_plot_'+dataset_name+'.png', dpi = 300.0)




    