This code is used to generate the results for the paper "Learning Axis-Aligned Decision Trees with Gradient Descent".

We provide two files, one jupyter notebook file with the config in the top cell and one python script where the config is loaded from "config.json". The python script only covers the main results, while the notebook shows additional results.

In the default setting we also lead the optimized hyperparameters for all approaches. This can also be adjusted in the config ("use_best_hpo_result").

By default, the GPU is used for the calculations of GDT. This can be changed in the config ("use_gpu"). If you receive a memory error for GPU computations, this is most likely due to the parallelization of the experiments on a single GPU. Please try to reduce the number of parallel computations ("n_jobs") or run on CPU ("use_gpu").
