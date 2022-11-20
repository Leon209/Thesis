



import argparse

parser = argparse.ArgumentParser(description="Evaluate Gradient-Based DTs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--use_cpu", action="store_true", help="use GPU for GDT calculation")
parser.add_argument("--gpu_num", help="GPU number")
parser.add_argument("--n_jobs", help="Parallel Jobs")
parser.add_argument("--default_params", action="store_true", help="Use Default Params (params from config) for evaluation")
args = parser.parse_args()
config_adjust = vars(args)

import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

import json
from numba import cuda       
import numpy as np
np.set_printoptions(suppress=True)

import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder

from livelossplot import PlotLosses

import os
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from IPython.display import Image
from IPython.display import display, clear_output

import pandas as pd




import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PYTHONWARNINGS"] = "ignore"
import logging

import tensorflow as tf
import tensorflow_addons as tfa

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

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

from itertools import product
from collections.abc import Iterable

from copy import deepcopy
from pathlib import Path
import pickle
import dill


from datetime import datetime



pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# In[3]:




# # Evaluation

# In[4]:
def main():

    install('tensorflow==2.9.1')
    install('tensorflow_addons')
    install('keras')
    install('sklearn')
    install('livelossplot')    
    install('tqdm')
    install('pandas')    
    install('joblib')
    install('dill')    
    install('pickle')
    install('seaborn')    
    install('scipy')
    install('genetic-tree')       
    install('numba')       
    install('tabulate')    
        
    with open('config.json') as f:
        config = json.load(f)

    for key, value in config_adjust.items():
        if key == 'use_cpu' and value is not None:
            config['computation']['use_gpu'] = not value
        if key == 'gpu_num' and value is not None:
            config['computation']['gpu_numbers'] = value
        if key == 'n_jobs' and value is not None:
            config['computation']['n_jobs'] = value
        if key == 'default_params' and value is not None:
            config['computation']['use_best_hpo_result'] = not value    



    if config['computation']['use_gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['computation']['gpu_numbers'])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.6"
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"    
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false' 
        
        
    tf.random.set_seed(config['computation']['random_seed'])
    np.random.seed(config['computation']['random_seed'])
    random.seed(config['computation']['random_seed'])          

    timestr = datetime.utcnow().strftime('%Y-%m-%d--%H-%M-%S%f')
    print(timestr)
    os.makedirs(os.path.dirname("./evaluation_results/latex_tables/" + timestr +"/"), exist_ok=True)

    filepath = './evaluation_results/depth' + str(config['gdt']['depth']) + '/' + timestr + '/'
    Path(filepath).mkdir(parents=True, exist_ok=True)    

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num XLA-GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))    
    
    #os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"     
    # ## Real-World Eval

    # ### Classification

    # In[6]:


    identifier_list_classification_binary = [
                            'BIN:Blood Transfusion',# 748 4
                            'BIN:Banknote Authentication',# 1372 4
                            'BIN:Titanic',# 891 7 
                            'BIN:Raisins',#900 7
                            'BIN:Rice',#3810 7
                            'BIN:Echocardiogram',#132 8 ###TD
                            'BIN:Wisconsin Diagnostic Breast Cancer',# 569 10
                            'BIN:Loan House',# 614 11
                            'BIN:Heart Failure',# 299 12
                            'BIN:Heart Disease',# 303 13
                            'BIN:Adult',# 32561 14
                            'BIN:Bank Marketing',# 45211 14
                            'BIN:Cervical Cancer',# 858 15
                            'BIN:Congressional Voting',# 435, 16 ###TD
                            'BIN:Absenteeism',# 740 18
                            'BIN:Hepatitis',#155 19 ###TD
                            'BIN:German',# 1000 20
                            'BIN:Mushroom',#8124 22
                            'BIN:Credit Card',# 30000 23
                            'BIN:Horse Colic',#368 27
                            'BIN:Thyroid',#9172 29 ###TD
                            'BIN:Spambase',# 4601 57
                      ]       


    # In[ ]:


    benchmark_dict = get_benchmark_dict(config=config, eval_identifier='classification') 

    parallel_eval_real_world = Parallel(n_jobs=min(config['computation']['n_jobs'], config['computation']['trials']), verbose=3, backend='loky') #loky #sequential multiprocessing
    evaluation_results_real_world_classification_binary = parallel_eval_real_world(delayed(evaluate_real_world_parallel_nested)(identifier_list=identifier_list_classification_binary[::-1], 
                                                                                                                               random_seed_data=config['computation']['random_seed']+i,
                                                                                                                               random_seed_model=config['computation']['random_seed'],
                                                                                                                               config = config,
                                                                                                                               benchmark_dict = benchmark_dict,
                                                                                                                               metrics = config['computation']['metrics_class'],
                                                                                                                               verbosity = -1) for i in range(config['computation']['trials']))


    # In[ ]:


    plot_table_save_results(benchmark_dict=benchmark_dict,
                            evaluation_results_real_world=evaluation_results_real_world_classification_binary,
                            identifier_list=identifier_list_classification_binary,                            
                            identifier_string='binary_test',
                            filepath=filepath,
                            config=config,
                            terminal_output=True,
                            plot_runtime=True)      
    if False:
        plot_table_save_results(benchmark_dict=benchmark_dict,
                                evaluation_results_real_world=evaluation_results_real_world_classification_binary,
                                identifier_list=identifier_list_classification_binary,                            
                                identifier_string='binary_valid',
                                filepath=filepath,
                                config=config,
                                terminal_output=True,
                                plot_runtime=True)  

        plot_table_save_results(benchmark_dict=benchmark_dict,
                                evaluation_results_real_world=evaluation_results_real_world_classification_binary,
                                identifier_list=identifier_list_classification_binary,                            
                                identifier_string='binary_train',
                                filepath=filepath,
                                config=config,
                                terminal_output=True,
                                plot_runtime=True)  


    # In[ ]:


    identifier_list_classification_multi = [
                            'MULT:Iris',# 150 4 3
                            'MULT:Balance Scale',# 625 4 3
                            'MULT:Car',# 1728 6 4
                            'MULT:Glass',# 214 9 6 
                            'MULT:Contraceptive',# 1473 9 3 
                            'MULT:Solar Flare',# 1389 10 8
                            'MULT:Wine',# 178 12 3
                            'MULT:Zoo',#101 16 7   ###TD
                            'MULT:Lymphography',# 148 18 4 ###TD
                            'MULT:Segment',# 2310 19 7
                            'MULT:Dermatology',# 366 34 6
                            'MULT:Landsat',# 6435 36 6
                            'MULT:Annealing',# 798 38 5
                            'MULT:Splice',# 3190 60 3
                      ]       


    # In[ ]:



    benchmark_dict = get_benchmark_dict(config=config, eval_identifier='classification')

    metrics = ['f1', 'roc_auc', 'accuracy']

    parallel_eval_real_world = Parallel(n_jobs=min(config['computation']['n_jobs'], config['computation']['trials']), verbose=3, backend='loky') #loky #sequential multiprocessing
    evaluation_results_real_world_classification_multi = parallel_eval_real_world(delayed(evaluate_real_world_parallel_nested)(identifier_list=identifier_list_classification_multi[::-1], 
                                                                                                           random_seed_data=config['computation']['random_seed']+i,
                                                                                                           random_seed_model=config['computation']['random_seed'],
                                                                                                           config = config,
                                                                                                           benchmark_dict = benchmark_dict,
                                                                                                           metrics = config['computation']['metrics_class'],
                                                                                                           verbosity = -1) for i in range(config['computation']['trials']))



    # In[ ]:


    plot_table_save_results(benchmark_dict=benchmark_dict,
                            evaluation_results_real_world=evaluation_results_real_world_classification_multi,
                            identifier_list=identifier_list_classification_multi,                            
                            identifier_string='multi_test',
                            filepath=filepath,
                            config=config,
                            terminal_output=True,
                            plot_runtime=True)      

    if False:
        plot_table_save_results(benchmark_dict=benchmark_dict,
                                evaluation_results_real_world=evaluation_results_real_world_classification_multi,
                                identifier_list=identifier_list_classification_multi,                            
                                identifier_string='multi_valid',
                                filepath=filepath,
                                config=config,
                                terminal_output=True,
                                plot_runtime=True)  

        plot_table_save_results(benchmark_dict=benchmark_dict,
                                evaluation_results_real_world=evaluation_results_real_world_classification_multi,
                                identifier_list=identifier_list_classification_multi,                            
                                identifier_string='multi_train',
                                filepath=filepath,
                                config=config,
                                terminal_output=True,
                                plot_runtime=True)  


    # In[ ]:


    if config['computation']['use_gpu']:
        device = cuda.get_current_device()
        device.reset()


    # In[ ]:




if __name__ == "__main__": 

        
	main()