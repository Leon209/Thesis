import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from typing import Callable

from livelossplot import PlotLosses

import os
import gc
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
from utilities.GDT import *


from joblib import Parallel, delayed
from gpuparallel import GPUParallel, delayed


from itertools import product
from collections.abc import Iterable

from copy import deepcopy

from keras.utils import np_utils

def make_batch(iterable, n=1, random_seed=42):
    tf.random.set_seed(random_seed)
    iterable = tf.random.shuffle(value=iterable, seed=random_seed)
    #rng = np.random.default_rng(seed=random_seed)
    #rng.shuffle(iterable)
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]   
        
def my_gather_nd(params, indices):
    idx_shape = tf.shape(indices)
    params_shape = tf.shape(params)
    idx_dims = idx_shape[-1]
    gather_shape = params_shape[idx_dims:]
    params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
    axis_step = tf.cast(tf.math.cumprod(params_shape[:idx_dims], exclusive=True, reverse=True), tf.int64)
    indices_flat = tf.reduce_sum(indices * axis_step, axis=-1)
    result_flat = tf.gather(params_flat, indices_flat)
    return tf.reshape(result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0))
        
def make_batch_det(iterable, n=1):
    l = iterable.shape[0]
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]   
        
def sigmoid(x, factor=1, shift_horizontal=0.0, shift_vertical=0.0):
    x = 1/(1+K.exp(-factor*(x-shift_horizontal))) + shift_vertical
    return x  

def tanh(x, factor=1, shift_horizontal=0, shift_vertical=0):
    x = (K.exp(factor*(x-shift_horizontal))-K.exp(-factor*(x-shift_horizontal)))/(K.exp(factor*(x-shift_horizontal))+K.exp(-factor*(x-shift_horizontal))) + shift_vertical
    return x 

@tf.custom_gradient
def round_with_gradients(x):
    def grad(dy):
        return dy
    return tf.round(x), grad


class GDT(tf.Module):
    
    def __init__(
            self,
            number_of_variables,
            number_of_classes,
        
            objective,
        
            loss,
        
            depth = 3,
        
            learning_rate_index = 1e-3,
            learning_rate_values = 1e-3,
            learning_rate_leaf = 1e-3,     
        
            optimizer = 'adam',
        
            dropout = 0.0,
        
            split_index_activation_beta = 1.0,
        
            split_index_activation = 'softmax',
        
            output_activation = 'softmax',
        
            initializer_values = 'GlorotUniform',
            initializer_index = 'GlorotUniform',
            initializer_leaf = 'GlorotUniform',
        
            normalize = None,
        
            initialized = False, #Flag to only initialzed everything one time
        
            random_seed = 42,
            verbosity = 1):   
         
        
        self.depth = depth
        
        self.objective = objective
        
        self.normalize = normalize
        
        self.learning_rate_index = learning_rate_index
        self.learning_rate_values = learning_rate_values
        self.learning_rate_leaf = learning_rate_leaf     
        self.optimizer = optimizer
        
        self.dropout = dropout
        
        self.split_index_activation_beta = split_index_activation_beta
        self.split_index_activation = split_index_activation
        self.output_activation = output_activation
        
        self.initializer_values = initializer_values
        self.initializer_index = initializer_index
        self.initializer_leaf = initializer_leaf
                
        self.seed = random_seed
        self.verbosity = verbosity
        self.number_of_variables = number_of_variables
        self.number_of_classes = number_of_classes

        
        self.internal_node_num_ = 2 ** self.depth - 1 
        self.leaf_node_num_ = 2 ** self.depth
        
        if loss == 'crossentropy':
            if self.number_of_classes == 2:
                self.loss = tf.keras.losses.get('binary_crossentropy')
            else:
                self.loss = tf.keras.losses.get('categorical_crossentropy')
        else:
            self.loss = tf.keras.losses.get(loss)                
        
        leaf_classes_array_shape = (self.leaf_node_num_,) if self.number_of_classes == 2 or self.objective == 'regression' else(self.leaf_node_num_, self.number_of_classes)
        
        tf.random.set_seed(self.seed)
        self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer_values, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)), trainable=True, name='split_values')
        
        tf.random.set_seed(self.seed)
        self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer_index, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)), trainable=True, name='split_index_array')            
         
        tf.random.set_seed(self.seed)
        self.leaf_classes_array = tf.Variable(tf.keras.initializers.get({'class_name': self.initializer_leaf, 'config': {'seed': self.seed}})(shape=leaf_classes_array_shape),trainable=True,name='leaf_classes_array')         

        self.optimizer_index = tf.keras.optimizers.get(self.optimizer)
        self.optimizer_values = tf.keras.optimizers.get(self.optimizer)
        self.optimizer_leaf = tf.keras.optimizers.get(self.optimizer)


        self.optimizer_index.learning_rate = self.learning_rate_index
        self.optimizer_values.learning_rate = self.learning_rate_values
        self.optimizer_leaf.learning_rate = self.learning_rate_leaf     
            
            
        self.plotlosses = PlotLosses()    
      
    
    
    def forward(self, 
                X: tf.Tensor, 
                training: bool):
        X = tf.dtypes.cast(tf.convert_to_tensor(X), tf.float32)               
                      
        split_index_array_complete = self.split_index_array
        split_values_complete = self.split_values

        split_index_array_complete = self.apply_dropout(split_index_array_complete,
                                                        training=training)
        
        if self.split_index_activation == 'softmax':
            split_index_array_complete = tf.keras.activations.softmax(self.split_index_activation_beta * split_index_array_complete)
        elif self.split_index_activation == 'entmax':
            split_index_array_complete = entmax15(self.split_index_activation_beta * split_index_array_complete)
        elif self.split_index_activation == 'sparsemax':
            split_index_array_complete = tfa.activations.sparsemax(self.split_index_activation_beta * split_index_array_complete)

        adjust_constant = tf.stop_gradient(split_index_array_complete -  tfa.seq2seq.hardmax(split_index_array_complete))
        split_index_array_complete = split_index_array_complete - adjust_constant        

        #split_index_array_complete_selected = tf.squeeze(tf.gather(split_index_array_complete, indices=tf.reshape(internal_node_index, (-1,1))))
        #split_values_complete_selected = tf.squeeze(tf.gather(split_values_complete, indices=tf.reshape(internal_node_index, (-1,1))))   
            
        
        #split_index_array_complete_selected = tf.squeeze(my_gather_nd(split_index_array_complete, indices=self.internal_node_index_list))
        #split_values_complete_selected = tf.squeeze(my_gather_nd(split_values_complete, indices=self.internal_node_index_list))           
        split_index_array_complete_selected = tf.squeeze(my_gather_nd(split_index_array_complete, indices=tf.reshape(self.internal_node_index_list, (-1,1))))
        split_values_complete_selected = tf.squeeze(my_gather_nd(split_values_complete, indices=tf.reshape(self.internal_node_index_list, (-1,1))))            

        
        split_index_array_complete_selected = tf.reshape(split_index_array_complete_selected, (self.leaf_node_num_,self.depth,self.number_of_variables))
        split_values_complete_selected = tf.reshape(split_values_complete_selected, (self.leaf_node_num_,self.depth,self.number_of_variables))
        s1 =  split_values_complete_selected * split_index_array_complete_selected
        s1_sum = tf.reduce_sum(s1, axis=2)
        s2 =  tf.expand_dims(tf.expand_dims(X, 1), 1) * split_index_array_complete_selected#split_index_array_complete[internal_node_index]
        s2_sum = tf.reduce_sum(s2, axis=3)
        node_result = tf.sigmoid(s1_sum-s2_sum)
        node_result_corrected = node_result - tf.stop_gradient(node_result - tf.round(node_result))

        p = tf.reduce_prod(((1-self.path_identifier_list)*node_result_corrected + self.path_identifier_list*(1-node_result_corrected)), axis=2)
        
        if self.objective == 'regression' or self.number_of_classes == 2:
            function_values_gdt = tf.reduce_sum(self.leaf_classes_array * p, axis=1)
        else:
            function_values_gdt = tf.reduce_sum(tf.expand_dims(p, 2) * self.leaf_classes_array, axis=1)      

        return function_values_gdt  

    @tf.function(jit_compile=True)
    def predict(self, X, batch_size = 2048, return_probabilities=False, denormalize=True):        
        
        pred_list = []
        for X_batch in list(make_batch_det(X, batch_size)):        
            preds_batch = self.forward(X_batch, 
                                 training=False)

            if self.objective == 'classification':
                if self.number_of_classes == 2:
                    preds_batch = tf.sigmoid(preds_batch)
                else:
                    if self.output_activation == "softmax":
                        if return_probabilities:
                            preds_batch = tf.keras.activations.softmax(preds_batch)
                        else:
                            preds_batch = tf.argmax(tf.keras.activations.softmax(preds_batch), axis=1)                
                    elif self.output_activation == "entmax":
                        if return_probabilities:
                            preds_batch = entmax15(preds_batch)
                        else:
                            preds_batch = tf.argmax(entmax15(preds_batch), axis=1)                    
                    elif self.output_activation == "sparsemax":
                        if return_probabilities:
                            preds_batch = tfa.activations.sparsemax(preds_batch)
                        else:
                            preds_batch = tf.argmax(tfa.activations.sparsemax(preds_batch), axis=1)                    
            elif self.objective == 'regression':
                if denormalize:
                    preds_batch = preds_batch = self.denormalize_labels(preds_batch)     
                    
            pred_list.append(preds_batch)
            
        preds = tf.concat(pred_list, axis=0)                 

        return preds 

    
    def backward(self, 
                 x: tf.Tensor,
                 y: tf.Tensor):
        
        
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape3:
                    tape1.watch(self.leaf_classes_array)
                    tape2.watch(self.split_values)
                    tape3.watch(self.split_index_array)

                    predicted = self.forward(x, training=True)

                    if self.objective == 'classification':
                        if self.number_of_classes == 2:
                            current_loss = tf.reduce_mean(self.loss(y, tf.sigmoid(predicted)))
                        else:                    
                            if self.output_activation == "softmax":
                                current_loss = tf.reduce_mean(self.loss(y, tf.keras.activations.softmax(predicted)))
                            elif self.output_activation == "entmax":
                                current_loss = tf.reduce_mean(self.loss(y, entmax15(predicted)))                   
                            elif self.output_activation == "sparsemax":
                                current_loss = tf.reduce_mean(self.loss(y, tfa.activations.sparsemax(predicted)))  
                    else:   
                        current_loss = tf.reduce_mean(self.loss(y, predicted))    
        
        grads1 = tape1.gradient(current_loss, self.leaf_classes_array)
        self.optimizer_leaf.apply_gradients(zip([grads1], [self.leaf_classes_array]))
        grads2 = tape2.gradient(current_loss, self.split_values)
        self.optimizer_values.apply_gradients(zip([grads2], [self.split_values]))
        grads3 = tape3.gradient(current_loss, self.split_index_array)
        self.optimizer_index.apply_gradients(zip([grads3], [self.split_index_array]))

        return current_loss, predicted
        
    
    
    def partial_fit(self, 
            X_train, 
            y_train, 
            
            batch_size=256, 
            epochs=1, 
            
            restarts = 0,
            restart_type='loss',#'metric'
            
            early_stopping_epochs=25, 
            early_stopping_type='loss',#'metric'
            early_stopping_epsilon = 0,
            
            valid_data=None,
            ):
         
        #Change to Partial fit method    
        if self.initialized == False:
            self.path_identifier_list = []
            self.internal_node_index_list = []
            for leaf_index in tf.unstack(tf.constant([i for i in range(self.leaf_node_num_)])):
                for current_depth in tf.unstack(tf.constant([i for i in range(1, self.depth+1)])):
                    path_identifier = tf.cast(tf.math.floormod(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-current_depth)))), 2), tf.float32)
                    internal_node_index =  tf.cast(tf.cast(tf.math.pow(2, (current_depth-1)), tf.float32) + tf.cast(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-(current_depth-1))))), tf.float32) - 1.0, tf.int64)
                    self.path_identifier_list.append(path_identifier)
                    self.internal_node_index_list.append(internal_node_index)
            self.path_identifier_list = tf.reshape(tf.stack(self.path_identifier_list), (-1,self.depth))
            self.internal_node_index_list = tf.reshape(tf.cast(tf.stack(self.internal_node_index_list), tf.int64), (-1,self.depth)) 
            
            
        #changing to tensors to increase efficiency
        X_train = tf.dtypes.cast(tf.convert_to_tensor(X_train), tf.float32)               
        y_train = tf.dtypes.cast(tf.convert_to_tensor(y_train), tf.float32)   
                
        if self.objective == 'classification':
            if self.number_of_classes >= 2:
                metric_name = 'f1'
                metric = tfa.metrics.F1Score(average='weighted', num_classes=self.number_of_classes)#tf.keras.metrics.CategoricalAccuracy()
            else:
                metric_name = 'f1'
                metric = tfa.metrics.F1Score(average='weighted', num_classes=self.number_of_classes)#tf.keras.metrics.BinaryAccuracy()
        elif self.objective == 'regression':
            metric_name = 'r2'
            metric = tfa.metrics.r_square.RSquare()        
        
        #valid data for stream learning none
        if valid_data is not None:
            valid_data = (tf.dtypes.cast(tf.convert_to_tensor(valid_data[0]), tf.float32), 
                          tf.dtypes.cast(tf.convert_to_tensor(valid_data[1]), tf.float32))
        
        self.data_mean = tf.cast(tf.math.reduce_mean(y_train), tf.float32)
        self.data_std = tf.cast(tf.math.reduce_std(y_train), tf.float32)     
        self.data_min = tf.cast(tf.math.reduce_min(y_train), tf.float32)
        self.data_max = tf.cast(tf.math.reduce_max(y_train), tf.float32)        
    
        #Normalize validation and y_train
        if self.objective == 'classification':
            if self.number_of_classes > 2 and (len(y_train.shape) == 1 or y_train.shape[1] == 1):
                if isinstance(y_train, pd.Series):
                    y_train = y_train.values
                y_train = np_utils.to_categorical(tf.reshape(y_train, (-1,1)), num_classes=self.number_of_classes) 

                if valid_data is not None:
                    valid_data_labels = valid_data[1]
                    if isinstance(valid_data_labels, pd.Series):
                        valid_data_labels = valid_data_labels.values  
                    valid_data_labels = np_utils.to_categorical(tf.reshape(valid_data_labels, (-1,1)), num_classes=self.number_of_classes)
                    valid_data = (valid_data[0], valid_data_labels)              
        else:   
            y_train = self.normalize_labels(y_train)   
            
            if valid_data is not None:
                valid_data = (valid_data[0], self.normalize_labels(valid_data[1]))
            
        #Change to Partial fit method    
        if self.initialized == False: 
            split_values_best_model = None#tf.identity(self.split_values)
            split_index_array_best_model = None#tf.identity(self.split_index_array)
            leaf_classes_array_best_model = None#tf.identity(self.leaf_classes_array)                    

            best_model_minimum_loss = np.inf
            best_model_minimum_metric = -np.inf
         #bis hier
            
        disable = True if self.verbosity == -1 else False
            
        for restart_number in tqdm(range(restarts+1), desc='restarts', disable=disable):
            
            backward_function = tf.function(self.backward, jit_compile=True)
            self.seed += restart_number
            
            
            #Change?
            if restart_number > 0:

                tf.keras.backend.clear_session()
                
                self.optimizer_index = tf.keras.optimizers.get(self.optimizer)
                self.optimizer_values = tf.keras.optimizers.get(self.optimizer)
                self.optimizer_leaf = tf.keras.optimizers.get(self.optimizer)

                self.optimizer_index.learning_rate = self.learning_rate_index
                self.optimizer_values.learning_rate = self.learning_rate_values
                self.optimizer_leaf.learning_rate = self.learning_rate_leaf                     
                
                tf.random.set_seed(self.seed)
                self.split_values.assign(tf.keras.initializers.get({'class_name': self.initializer_values, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)))

                self.split_index_array.assign(tf.keras.initializers.get({'class_name': self.initializer_index, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)))

                leaf_classes_array_shape = (self.leaf_node_num_,) if self.number_of_classes == 2 or self.objective == 'regression' else(self.leaf_node_num_, self.number_of_classes)

                self.leaf_classes_array.assign(tf.keras.initializers.get({'class_name': self.initializer_leaf, 'config': {'seed': self.seed}})(shape=leaf_classes_array_shape))                  
                
            minimum_loss_epoch = np.inf
            minimum_loss_epoch_valid = np.inf 

            minimum_metric_epoch = -np.inf
            minimum_metric_epoch_valid = -np.inf

            epochs_without_improvement = 0    

            batch_size = min(batch_size, int(np.ceil(X_train.shape[0]/2)))
            #shuffle data
            #Hier Kann mein Code eingefügt werden
            for current_epoch in tqdm(range(epochs), desc='epochs', disable=disable):                
                tf.random.set_seed(self.seed + current_epoch)
                X_train_epoch = tf.random.shuffle(X_train, seed=self.seed + current_epoch)
                tf.random.set_seed(self.seed + current_epoch)
                y_train_epoch = tf.random.shuffle(y_train, seed=self.seed + current_epoch)      

                loss_list = []   
                preds_list = []


                for index, (X_batch, y_batch) in enumerate(zip(make_batch_det(X_train_epoch, batch_size), make_batch_det(y_train_epoch, batch_size))):
                    current_loss, preds_batch = backward_function(X_batch, y_batch)#self.backward(X_batch, y_batch)
                    loss_list.append(float(current_loss))
                    preds_list.append(preds_batch)
                    if self.verbosity > 2:
                        batch_idx = (index+1)*batch_size
                        msg = "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                        print(msg.format(current_epoch, batch_idx, current_loss))                   

                current_loss_epoch = np.mean(loss_list)
                preds = tf.concat(preds_list, axis=0) 

                loss_dict = {'loss': current_loss_epoch}

                preds_metric = self.adjust_preds_for_metric(preds, logits=True)
                if self.number_of_classes == 2:
                    loss_dict[metric_name] = tf.reduce_mean(metric(np_utils.to_categorical(tf.reshape(y_train_epoch, (-1,1)), num_classes=self.number_of_classes), np_utils.to_categorical(tf.reshape(preds_metric, (-1,1)), num_classes=self.number_of_classes)))                 
                else:
                    loss_dict[metric_name] = tf.reduce_mean(metric(y_train_epoch, preds_metric))                 

                if valid_data is not None:
                    preds_val = self.predict(valid_data[0], return_probabilities=True, denormalize=False)#self.forward(valid_data[0], training=False)     

                    current_loss_epoch_valid = tf.reduce_mean(self.loss(valid_data[1], preds_val))
                    loss_dict['val_loss'] = current_loss_epoch_valid   
                    preds_val_metric = self.adjust_preds_for_metric(preds_val)
                    if self.number_of_classes == 2:
                        loss_dict['val_' + metric_name] = tf.reduce_mean(metric(np_utils.to_categorical(tf.reshape(valid_data[1], (-1,1)), num_classes=self.number_of_classes), np_utils.to_categorical(tf.reshape(preds_val_metric, (-1,1)), num_classes=self.number_of_classes)))                 
                    else:
                        loss_dict['val_' + metric_name] = tf.reduce_mean(metric(valid_data[1], preds_val_metric))                    


                if self.verbosity > 1:    
                    msg = "Epoch: {:02d} | Loss: {:.5f} |"
                    print(msg.format(current_epoch, current_loss_epoch))              
                    if valid_data is not None:
                        msg = "Epoch: {:02d} | Valid Loss: {:.5f} |"
                        print(msg.format(current_epoch, current_loss_epoch_valid))                   

                if self.verbosity == 1:  


                    self.plotlosses.update(loss_dict)
                    self.plotlosses.send()            

                if early_stopping_type == 'metric' or restart_type == 'metric': 

                    if valid_data is not None:
                        if loss_dict['val_' + metric_name] - early_stopping_epsilon > minimum_metric_epoch_valid:
                            minimum_metric_epoch_valid = loss_dict['val_' + metric_name]#current_loss_epoch_valid
                            if early_stopping_type == 'metric':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)          

                        else:
                            if early_stopping_type == 'metric':
                                epochs_without_improvement += 1            
                    else:
                        if loss_dict[metric_name] - early_stopping_epsilon > minimum_metric_epoch:
                            minimum_metric_epoch = loss_dict[metric_name]#current_loss_epoch

                            if early_stopping_type == 'metric':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)       

                        else:
                            if early_stopping_type == 'metric':
                                epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_epochs:  
                        try:
                            self.split_values.assign(split_values_stored)
                            self.split_index_array.assign(split_index_array_stored)
                            self.leaf_classes_array.assign(leaf_classes_array_stored)   
                        except UnboundLocalError:
                            pass

                        break                   
                if early_stopping_type == 'loss' or restart_type == 'loss': 
                    if valid_data is not None:
                        if current_loss_epoch_valid + early_stopping_epsilon < minimum_loss_epoch_valid:
                            minimum_loss_epoch_valid = current_loss_epoch_valid

                            if early_stopping_type == 'loss':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)       

                        else:
                            if early_stopping_type == 'loss':
                                epochs_without_improvement += 1            
                    else:
                        if current_loss_epoch + early_stopping_epsilon < minimum_loss_epoch:
                            minimum_loss_epoch = current_loss_epoch

                            if early_stopping_type == 'loss':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)       

                        else:
                            if early_stopping_type == 'loss':
                                epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_epochs:  
                        try:
                            self.split_values.assign(split_values_stored)
                            self.split_index_array.assign(split_index_array_stored)
                            self.leaf_classes_array.assign(leaf_classes_array_stored)   
                        except UnboundLocalError:
                            pass
                        break
                        

            if valid_data is not None:
                if restart_type == 'metric':                
                    if minimum_metric_epoch_valid > best_model_minimum_metric or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:                     
                        
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_metric = minimum_metric_epoch_valid            

                else:
                    if minimum_loss_epoch_valid < best_model_minimum_loss or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:             
                        
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_loss = minimum_loss_epoch_valid                           
            else:
                if restart_type == 'metric': 
                    if minimum_metric_epoch > best_model_minimum_metric or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_metric = minimum_metric_epoch              

                else:
                    if minimum_loss_epoch < best_model_minimum_loss or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_loss = minimum_loss_epoch       
        
        try:
            self.split_values.assign(split_values_best_model)
            self.split_index_array.assign(split_index_array_best_model)
            self.leaf_classes_array.assign(leaf_classes_array_best_model)   
        except UnboundLocalError:
            pass
        
    def fit(self, 
            X_train, 
            y_train, 
            
            batch_size=256, 
            epochs=1, 
            
            restarts = 0,
            restart_type='loss',#'metric'
            
            early_stopping_epochs=25, 
            early_stopping_type='loss',#'metric'
            early_stopping_epsilon = 0,
            
            valid_data=None,
            ):
         
        #überflüssig
        self.path_identifier_list = []
        self.internal_node_index_list = []
        for leaf_index in tf.unstack(tf.constant([i for i in range(self.leaf_node_num_)])):
            for current_depth in tf.unstack(tf.constant([i for i in range(1, self.depth+1)])):
                path_identifier = tf.cast(tf.math.floormod(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-current_depth)))), 2), tf.float32)
                internal_node_index =  tf.cast(tf.cast(tf.math.pow(2, (current_depth-1)), tf.float32) + tf.cast(tf.math.floor(leaf_index/(tf.math.pow(2, (self.depth-(current_depth-1))))), tf.float32) - 1.0, tf.int64)
                self.path_identifier_list.append(path_identifier)
                self.internal_node_index_list.append(internal_node_index)
        self.path_identifier_list = tf.reshape(tf.stack(self.path_identifier_list), (-1,self.depth))
        self.internal_node_index_list = tf.reshape(tf.cast(tf.stack(self.internal_node_index_list), tf.int64), (-1,self.depth))            #bis hier
            
        #changing to tensors to increase efficiency
        X_train = tf.dtypes.cast(tf.convert_to_tensor(X_train), tf.float32)               
        y_train = tf.dtypes.cast(tf.convert_to_tensor(y_train), tf.float32)   
                
        if self.objective == 'classification':
            if self.number_of_classes >= 2:
                metric_name = 'f1'
                metric = tfa.metrics.F1Score(average='weighted', num_classes=self.number_of_classes)#tf.keras.metrics.CategoricalAccuracy()
            else:
                metric_name = 'f1'
                metric = tfa.metrics.F1Score(average='weighted', num_classes=self.number_of_classes)#tf.keras.metrics.BinaryAccuracy()
        elif self.objective == 'regression':
            metric_name = 'r2'
            metric = tfa.metrics.r_square.RSquare()        
        
        #valid data for stream learning none
        if valid_data is not None:
            valid_data = (tf.dtypes.cast(tf.convert_to_tensor(valid_data[0]), tf.float32), 
                          tf.dtypes.cast(tf.convert_to_tensor(valid_data[1]), tf.float32))
        
        self.data_mean = tf.cast(tf.math.reduce_mean(y_train), tf.float32)
        self.data_std = tf.cast(tf.math.reduce_std(y_train), tf.float32)     
        self.data_min = tf.cast(tf.math.reduce_min(y_train), tf.float32)
        self.data_max = tf.cast(tf.math.reduce_max(y_train), tf.float32)        
    
        #Normalize validation and y_train
        if self.objective == 'classification':
            if self.number_of_classes > 2 and (len(y_train.shape) == 1 or y_train.shape[1] == 1):
                if isinstance(y_train, pd.Series):
                    y_train = y_train.values
                y_train = np_utils.to_categorical(tf.reshape(y_train, (-1,1)), num_classes=self.number_of_classes) 

                if valid_data is not None:
                    valid_data_labels = valid_data[1]
                    if isinstance(valid_data_labels, pd.Series):
                        valid_data_labels = valid_data_labels.values  
                    valid_data_labels = np_utils.to_categorical(tf.reshape(valid_data_labels, (-1,1)), num_classes=self.number_of_classes)
                    valid_data = (valid_data[0], valid_data_labels)              
        else:   
            y_train = self.normalize_labels(y_train)   
            
            if valid_data is not None:
                valid_data = (valid_data[0], self.normalize_labels(valid_data[1]))
            
            
        split_values_best_model = None#tf.identity(self.split_values)
        split_index_array_best_model = None#tf.identity(self.split_index_array)
        leaf_classes_array_best_model = None#tf.identity(self.leaf_classes_array)                    
            
        best_model_minimum_loss = np.inf
        best_model_minimum_metric = -np.inf
            
        disable = True if self.verbosity == -1 else False
            
        for restart_number in tqdm(range(restarts+1), desc='restarts', disable=disable):
            
            backward_function = tf.function(self.backward, jit_compile=True)
            self.seed += restart_number
            
            
            if restart_number > 0:

                tf.keras.backend.clear_session()
                
                self.optimizer_index = tf.keras.optimizers.get(self.optimizer)
                self.optimizer_values = tf.keras.optimizers.get(self.optimizer)
                self.optimizer_leaf = tf.keras.optimizers.get(self.optimizer)

                self.optimizer_index.learning_rate = self.learning_rate_index
                self.optimizer_values.learning_rate = self.learning_rate_values
                self.optimizer_leaf.learning_rate = self.learning_rate_leaf                     
                
                tf.random.set_seed(self.seed)
                self.split_values.assign(tf.keras.initializers.get({'class_name': self.initializer_values, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)))

                self.split_index_array.assign(tf.keras.initializers.get({'class_name': self.initializer_index, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)))

                leaf_classes_array_shape = (self.leaf_node_num_,) if self.number_of_classes == 2 or self.objective == 'regression' else(self.leaf_node_num_, self.number_of_classes)

                self.leaf_classes_array.assign(tf.keras.initializers.get({'class_name': self.initializer_leaf, 'config': {'seed': self.seed}})(shape=leaf_classes_array_shape))                  
                
            minimum_loss_epoch = np.inf
            minimum_loss_epoch_valid = np.inf 

            minimum_metric_epoch = -np.inf
            minimum_metric_epoch_valid = -np.inf

            epochs_without_improvement = 0    

            batch_size = min(batch_size, int(np.ceil(X_train.shape[0]/2)))
            #shuffle data
            #Hier Kann mein Code eingefügt werden
            for current_epoch in tqdm(range(epochs), desc='epochs', disable=disable):                
                tf.random.set_seed(self.seed + current_epoch)
                X_train_epoch = tf.random.shuffle(X_train, seed=self.seed + current_epoch)
                tf.random.set_seed(self.seed + current_epoch)
                y_train_epoch = tf.random.shuffle(y_train, seed=self.seed + current_epoch)      

                loss_list = []   
                preds_list = []


                for index, (X_batch, y_batch) in enumerate(zip(make_batch_det(X_train_epoch, batch_size), make_batch_det(y_train_epoch, batch_size))):
                    current_loss, preds_batch = backward_function(X_batch, y_batch)#self.backward(X_batch, y_batch)
                    loss_list.append(float(current_loss))
                    preds_list.append(preds_batch)
                    if self.verbosity > 2:
                        batch_idx = (index+1)*batch_size
                        msg = "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                        print(msg.format(current_epoch, batch_idx, current_loss))                   

                current_loss_epoch = np.mean(loss_list)
                preds = tf.concat(preds_list, axis=0) 

                loss_dict = {'loss': current_loss_epoch}

                preds_metric = self.adjust_preds_for_metric(preds, logits=True)
                if self.number_of_classes == 2:
                    loss_dict[metric_name] = tf.reduce_mean(metric(np_utils.to_categorical(tf.reshape(y_train_epoch, (-1,1)), num_classes=self.number_of_classes), np_utils.to_categorical(tf.reshape(preds_metric, (-1,1)), num_classes=self.number_of_classes)))                 
                else:
                    loss_dict[metric_name] = tf.reduce_mean(metric(y_train_epoch, preds_metric))                 

                if valid_data is not None:
                    preds_val = self.predict(valid_data[0], return_probabilities=True, denormalize=False)#self.forward(valid_data[0], training=False)     

                    current_loss_epoch_valid = tf.reduce_mean(self.loss(valid_data[1], preds_val))
                    loss_dict['val_loss'] = current_loss_epoch_valid   
                    preds_val_metric = self.adjust_preds_for_metric(preds_val)
                    if self.number_of_classes == 2:
                        loss_dict['val_' + metric_name] = tf.reduce_mean(metric(np_utils.to_categorical(tf.reshape(valid_data[1], (-1,1)), num_classes=self.number_of_classes), np_utils.to_categorical(tf.reshape(preds_val_metric, (-1,1)), num_classes=self.number_of_classes)))                 
                    else:
                        loss_dict['val_' + metric_name] = tf.reduce_mean(metric(valid_data[1], preds_val_metric))                    


                if self.verbosity > 1:    
                    msg = "Epoch: {:02d} | Loss: {:.5f} |"
                    print(msg.format(current_epoch, current_loss_epoch))              
                    if valid_data is not None:
                        msg = "Epoch: {:02d} | Valid Loss: {:.5f} |"
                        print(msg.format(current_epoch, current_loss_epoch_valid))                   

                if self.verbosity == 1:  


                    self.plotlosses.update(loss_dict)
                    self.plotlosses.send()            

                if early_stopping_type == 'metric' or restart_type == 'metric': 

                    if valid_data is not None:
                        if loss_dict['val_' + metric_name] - early_stopping_epsilon > minimum_metric_epoch_valid:
                            minimum_metric_epoch_valid = loss_dict['val_' + metric_name]#current_loss_epoch_valid
                            if early_stopping_type == 'metric':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)          

                        else:
                            if early_stopping_type == 'metric':
                                epochs_without_improvement += 1            
                    else:
                        if loss_dict[metric_name] - early_stopping_epsilon > minimum_metric_epoch:
                            minimum_metric_epoch = loss_dict[metric_name]#current_loss_epoch

                            if early_stopping_type == 'metric':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)       

                        else:
                            if early_stopping_type == 'metric':
                                epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_epochs:  
                        try:
                            self.split_values.assign(split_values_stored)
                            self.split_index_array.assign(split_index_array_stored)
                            self.leaf_classes_array.assign(leaf_classes_array_stored)   
                        except UnboundLocalError:
                            pass

                        break                   
                if early_stopping_type == 'loss' or restart_type == 'loss': 
                    if valid_data is not None:
                        if current_loss_epoch_valid + early_stopping_epsilon < minimum_loss_epoch_valid:
                            minimum_loss_epoch_valid = current_loss_epoch_valid

                            if early_stopping_type == 'loss':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)       

                        else:
                            if early_stopping_type == 'loss':
                                epochs_without_improvement += 1            
                    else:
                        if current_loss_epoch + early_stopping_epsilon < minimum_loss_epoch:
                            minimum_loss_epoch = current_loss_epoch

                            if early_stopping_type == 'loss':
                                epochs_without_improvement = 0
                                split_values_stored = tf.identity(self.split_values)
                                split_index_array_stored = tf.identity(self.split_index_array)
                                leaf_classes_array_stored = tf.identity(self.leaf_classes_array)       

                        else:
                            if early_stopping_type == 'loss':
                                epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_epochs:  
                        try:
                            self.split_values.assign(split_values_stored)
                            self.split_index_array.assign(split_index_array_stored)
                            self.leaf_classes_array.assign(leaf_classes_array_stored)   
                        except UnboundLocalError:
                            pass
                        break
                        

            if valid_data is not None:
                if restart_type == 'metric':                
                    if minimum_metric_epoch_valid > best_model_minimum_metric or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:                     
                        
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_metric = minimum_metric_epoch_valid            

                else:
                    if minimum_loss_epoch_valid < best_model_minimum_loss or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:             
                        
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_loss = minimum_loss_epoch_valid                           
            else:
                if restart_type == 'metric': 
                    if minimum_metric_epoch > best_model_minimum_metric or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_metric = minimum_metric_epoch              

                else:
                    if minimum_loss_epoch < best_model_minimum_loss or restart_number == 0:# best_model_minimum_metric > best_model_minimum_metric:
                        split_values_best_model = tf.identity(self.split_values)
                        split_index_array_best_model = tf.identity(self.split_index_array)
                        leaf_classes_array_best_model = tf.identity(self.leaf_classes_array)                    

                        best_model_minimum_loss = minimum_loss_epoch       
        
        try:
            self.split_values.assign(split_values_best_model)
            self.split_index_array.assign(split_index_array_best_model)
            self.leaf_classes_array.assign(leaf_classes_array_best_model)   
        except UnboundLocalError:
            pass
        
                   
    
    
    
    
    
    
    def set_params(self, **kwargs):
        
        #print(kwargs)
        
        conditional_arguments = ['random_seed',
                                'depth',
                                'initializer_values',
                                'initializer_index',
                                'initializer_leaf',
                                'optimizer',
                                'learning_rate_index',
                                'learning_rate_values',
                                'learning_rate_leaf',     
                                'loss']
        
        excluded_arguments = ['batch_size',
                              'epochs',
                              'restarts',
                              'restart_type',
                              'early_stopping_epochs',
                              'early_stopping_type',
                             ]       
        
        for arg_key, arg_value in kwargs.items():
            if arg_key not in conditional_arguments and arg_key not in excluded_arguments:
                setattr(self, arg_key, arg_value)
        
        for conditional_argument in conditional_arguments:
            arg_key = conditional_argument
            if conditional_argument in kwargs.keys():
                arg_value = kwargs[arg_key]
            else:
                if arg_key != 'random_seed':
                    arg_value = self.__dict__[arg_key]
                    #setattr(self, arg_key, arg_value)
                else:
                    arg_value = self.__dict__['seed']
                    #setattr(self, 'seed', arg_value)
                
            if arg_key == 'random_seed':
                self.seed = arg_value
            elif arg_key == 'depth':
                self.depth = arg_value
                self.internal_node_num_ = 2 ** self.depth - 1 
                self.leaf_node_num_ = 2 ** self.depth            
            elif arg_key == 'initializer_values':
                self.initializer_values = arg_value      
                tf.random.set_seed(self.seed)
                self.split_values = tf.Variable(tf.keras.initializers.get({'class_name': arg_value, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)),trainable=True,name='split_values')                
            elif arg_key == 'initializer_index':
                self.initializer_index = arg_value
                tf.random.set_seed(self.seed)
                self.split_index_array = tf.Variable(tf.keras.initializers.get({'class_name': arg_value, 'config': {'seed': self.seed}})(shape=(self.internal_node_num_, self.number_of_variables)),trainable=True,name='split_index_array')                        
            elif arg_key == 'initializer_leaf':
                self.initializer_leaf = arg_value                          
                tf.random.set_seed(self.seed)
                leaf_classes_array_shape = (self.leaf_node_num_,) if self.number_of_classes == 2 or self.objective == 'regression' else(self.leaf_node_num_, self.number_of_classes)
                self.leaf_classes_array = tf.Variable(tf.keras.initializers.get({'class_name': arg_value, 'config': {'seed': self.seed}})(shape=leaf_classes_array_shape), trainable=True,name='leaf_classes_array')                     
            elif arg_key == 'optimizer':
                self.optimizer = arg_value
                self.optimizer_index = tf.keras.optimizers.get(arg_value)
                self.optimizer_values = tf.keras.optimizers.get(arg_value)
                self.optimizer_leaf = tf.keras.optimizers.get(arg_value)                         
            elif arg_key == 'learning_rate_index':
                self.learning_rate_index = arg_value                
                self.optimizer_index.learning_rate = arg_value
            elif arg_key == 'learning_rate_values':
                self.learning_rate_values = arg_value
                self.optimizer_values.learning_rate = arg_value
            elif arg_key == 'learning_rate_leaf':
                self.learning_rate_leaf = arg_value   
                self.optimizer_leaf.learning_rate = arg_value
            elif arg_key == 'loss':
                if arg_value == 'crossentropy':
                    if self.number_of_classes == 2:
                        self.loss = tf.keras.losses.get('binary_crossentropy')
                    else:
                        self.loss = tf.keras.losses.get('categorical_crossentropy')
                else:
                    self.loss = tf.keras.losses.get(arg_value)                  

    
    def adjust_preds_for_metric(self, preds, logits=False, denormalize=True):
        
        if logits:
            if self.objective == 'classification':
                if self.number_of_classes == 2:
                    preds = tf.sigmoid(preds)
                else:
                    if self.output_activation == "softmax":
                        preds = tf.keras.activations.softmax(preds)             
                    elif self.output_activation == "entmax":
                        preds = entmax15(preds)                  
                    elif self.output_activation == "sparsemax":
                        preds = tfa.activations.sparsemax(preds)                
            elif self.objective == 'regression':
                if denormalize:
                    preds = preds_batch = self.denormalize_labels(preds)             
        
        
        if self.objective == 'classification':
            if self.number_of_classes == 2:
                preds_metric = tf.round(preds)
            else:
                preds_metric = tfa.seq2seq.hardmax(preds)  
        if self.objective == 'regression':
            preds_metric = preds    
            
        return preds_metric
    
    


    def normalize_labels(self,
                        labels: tf.Tensor):
        if self.normalize == 'mean':
            labels = (labels - self.data_mean) / self.data_std
        elif self.normalize == 'min-max':
            labels = (labels - self.data_min) / (self.data_max - self.data_min)

        return labels

    def denormalize_labels(self,
                           labels: tf.Tensor):
        if self.normalize == 'mean':
            labels = labels * self.data_std + self.data_mean
        elif self.normalize == 'min-max':
            labels = labels * (self.data_max - self.data_min) + self.data_min    

        return labels

    def apply_dropout(self,
                      index_array: tf.Tensor,
                      training: bool):

        if training and self.dropout > 0.0:
            row_index = tf.cast(tf.constant([i for i in range(index_array.shape[0])]), tf.int64)
            
            tf.random.set_seed(self.seed)
            mask = tf.cast(tf.greater(np.random.uniform(0,1,(index_array.shape[0])), self.dropout), tf.float32)

            max_split = tf.stop_gradient(tf.argmax(index_array, axis=1))
            max_split_with_index = tf.transpose(tf.stack([row_index,max_split]))#tf.concat([row_index,max_split], axis=1)
            sparse_tensor = tf.SparseTensor(
                                  dense_shape=index_array.shape,#[index_array.shape[0], index_array.shape[1]],
                                  values=[0.0 for i in range(index_array.shape[0])],
                                  indices = max_split_with_index)
            dense_tensor = tf.sparse.to_dense(sparse_tensor, default_value = 1.0)

            dropout_mask = tf.cast(tf.greater(tf.math.add(dense_tensor, tf.expand_dims(mask, 1)), 0), tf.float32)

            index_array = index_array * dropout_mask

        return index_array        

        
        

    def plot(self, normalizer_list=None, path='./dt_plot.png'):
        from anytree import Node, RenderTree
        from anytree.exporter import DotExporter

        split_index_list_by_internal_node_max = tfa.seq2seq.hardmax(self.split_index_array)
        split_values_list_by_internal_node = self.split_values
 
        splits = tf.stack(tf.multiply(split_values_list_by_internal_node, split_index_list_by_internal_node_max))
        
        splits = splits.numpy()
        
        if self.objective == 'classification':
            if self.number_of_classes == 2:
                leaf_classes = tf.sigmoid(self.leaf_classes_array).numpy()
            else:                    
                if self.output_activation == "softmax":
                    leaf_classes = tf.keras.activations.softmax(self.leaf_classes_array).numpy()
                elif self.output_activation == "entmax":
                    leaf_classes = entmax15(self.leaf_classes_array).numpy()
                elif self.output_activation == "sparsemax":
                    leaf_classes = tfa.activations.sparsemax(self.leaf_classes_array).numpy()
        else:  
            leaf_classes = self.leaf_classes_array.numpy()
            if self.normalize is not None:
                leaf_classes = self.denormalize_labels(leaf_classes)            
        


        if normalizer_list is not None: 
            transpose = splits.transpose()
            transpose_normalized = []
            for i, column in enumerate(transpose):
                column_new = column
                if len(column_new[column_new != 0]) != 0:
                    column_new[column_new != 0] = normalizer_list[i].inverse_transform(column[column != 0].reshape(-1, 1)).ravel()
                #column_new = normalizer_list[i].inverse_transform(column.reshape(-1, 1)).ravel()
                transpose_normalized.append(column_new)
            splits = np.array(transpose_normalized).transpose()

          
            
        splits_by_layer = []
        for i in range(self.depth+1):
            start = 2**i - 1
            end = 2**(i+1) -1
            splits_by_layer.append(splits[start:end])

        nodes = {
        }
        #tree = Tree()
        for i, splits in enumerate(splits_by_layer):
            for j, split in enumerate(splits):
                if i == 0:
                    current_node_id = int(2**i - 1 + j)
                    name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                    split_variable = np.argmax(np.abs(split))
                    split_value = np.round(split[split_variable], 3)
                    split_description = 'x' + str(split_variable) + ' >= '  + str(split_value)

                    nodes[name] = Node(name=name, display_name=split_description)

                    #tree.create_node(tag=split_description, identifier=name, data=None)            
                else:
                    current_node_id = int(2**i - 1 + j)
                    name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
                    parent_node_id = int(np.floor((current_node_id-1)/2))
                    parent_name = 'n' + str(parent_node_id)
                    split_variable = np.argmax(np.abs(split))
                    split_value = np.round(split[split_variable], 3)
                    split_description = 'x' + str(split_variable) + ' >= '  + str(split_value)

                    nodes[name] = Node(name=name, parent=nodes[parent_name], display_name=split_description)

                    #tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)

        for j, leaf_class in enumerate(leaf_classes):
            i = self.depth
            current_node_id = int(2**i - 1 + j)
            name = 'n' + str(current_node_id)#'l' + str(i) + 'n' + str(j)
            parent_node_id = int(np.floor((current_node_id-1)/2))
            parent_name = 'n' + str(parent_node_id)
            #split_variable = np.argmax(np.abs(split))
            #split_value = np.round(split[split_variable], 3)
            split_description = str(np.round((leaf_class), 3))#'x' + str(split_variable) + ' <= '  + str(split_value)
            nodes[name] = Node(name=name, parent=nodes[parent_name], display_name=split_description)
            #tree.create_node(tag=split_description, identifier=name, parent=parent_name, data=None)        

            DotExporter(nodes['n0'], nodeattrfunc=lambda node: 'label="{}"'.format(node.display_name)).to_picture(path)


        return Image(path)#, nodes#nodes#tree        

        
        
def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """
    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True) 
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs
    
        return outputs, grad_fn
    
    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param labels: reference answers vector int64[batch_size] \in [0, num_classes)
    :param logits: output matrix float32[batch_size, num_classes] (not actually logits :)
    :returns: elementwise loss, float32[batch_size]
    """
    assert logits.shape.ndims == 2 and labels.shape.ndims == 1
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)
    
    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def entmax15_loss_with_logits(labels, logits):
    """
    Computes sample-wise entmax1.5 loss
    :param logits: "logits" matrix float32[batch_size, num_classes]
    :param labels: reference answers indicators, float32[batch_size, num_classes]
    :returns: elementwise loss, float32[batch_size]
    
    WARNING: this function does not propagate gradients through :labels:
    This behavior is the same as like softmax_crossentropy_with_logits v1
    It may become an issue if you do something like co-distillation
    """
    assert labels.shape.ndims == logits.shape.ndims == 2
    with tf.name_scope('entmax_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('entmax_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """ performs tf.nn.top_k over any chosen axis """
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """
    replicates the behavior of torch.gather for tf<=1.8;
    for newer versions use tf.gather with batch_dims
    :param values: tensor [d0, ..., dn]
    :param indices: int64 tensor of same shape as values except for gather_axis
    :param gather_axis: performs gather along this axis
    :returns: gathered values, same shape as values except for gather_axis
        If gather_axis == 2
        gathered_values[i, j, k, ...] = values[i, j, indices[i, j, k, ...], ...]
        see torch.gather for more detils
    """
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_support'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size


