def get_config_for_dataset(dataset_name)-> {}:
    if(dataset_name == 'agr_g'):
        config = {
            'gdt': {
                'depth': 9,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.5, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0.95,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 10000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    if(dataset_name == 'agr_a'):
        config = {
            'gdt': {
                'depth': 9,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.2, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0.5,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 10000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    
    
    if(dataset_name == 'airlines'):
        config = {
            'gdt': {
                'depth': 11,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.2, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 10000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    if(dataset_name == 'electricity'):
        config = {
            'gdt': {
                'depth': 5,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 128,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0.5,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 1000,#default 200
                'pretrain_size': 1000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    
    if(dataset_name == 'hyperplane'):
        config = {
            'gdt': {
                'depth': 10,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.2, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0.5,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 10000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    if(dataset_name == 'NOAA_Weather'):
        config = {
            'gdt': {
                'depth': 5,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.2, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 128,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 20,
            },

            'preprocessing': {
                'balance_threshold': 0.95,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 1000,#default 200
                'pretrain_size': 1000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    if(dataset_name == 'rbf_f'):
        config = {
            'gdt': {
                'depth': 11,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.2, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 250000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 5000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    if(dataset_name == 'rbf_m'):
        config = {
            'gdt': {
                'depth': 11,

                'learning_rate_index': 0.02,
                'learning_rate_values': 0.003,
                'learning_rate_leaf': 0.002,

                'dropout': 0.5, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 250000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 2500,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    
    if(dataset_name == 'sea_a'):
        config = {
            'gdt': {
                'depth': 11,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.2, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 10000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    if(dataset_name == 'sea_g'):
        config = {
            'gdt': {
                'depth': 11,

                'learning_rate_index': 0.05,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.005,

                'dropout': 0.2, #0.2 oder 0.5


                'initializer_values': 'GlorotUniform', 
                'initializer_index': 'GlorotUniform', 
                'initializer_leaf': 'GlorotUniform', 

                'optimizer': 'adam', 

                'batch_size': 256,#120
                'epochs': 1,

                'restarts': 0,#
                'restart_type': 'loss', #'loss', 'metric'

                'early_stopping_epochs': 600,
                'early_stopping_type': 'loss', #'loss', 'metric'
                'early_stopping_epsilon': 0.0,

                'pretrain_epochs': 30,
            },

            'preprocessing': {
                'balance_threshold': 0,#.25, #if minclass fraction less than threshold/num_classes | #0=no rebalance, 1=rebalance all
                'normalization_technique': 'mean', #'min-max'
            },

            'computation': {
                'random_seed': 42,
                'trials': 11, # fixed to 1 for HPO

                'use_best_hpo_result': True,
                'force_depth': False,

                'use_gpu': True,
                'gpu_numbers': '3',#'1',
                'n_jobs': 10, #vorher 20
                'verbosity': 0,


                'hpo': None,#'binary', #'binary', 'multi', 'regression'
                'search_iterations': 300,
                'cv_num': 3,     

                'metrics_class': ['f1', 'roc_auc', 'accuracy'],

                'metrics_reg': ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'],

                'eval_metric_class': ['f1', 'roc_auc'], #f1 accuracy
                'eval_metric_reg': 'r2', #r2 mae        

                'max_total_samples': 1000000,
                'chunk_size': 2500,#default 200
                'pretrain_size': 10000,
            },

            'benchmarks': {       
            }
        }
        return config
    
    
    
    