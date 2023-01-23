# GDTs for Stream Learning

We provide 5 notebooks
+ dataset_generation and main_evaluation are used to create most of our results.

+ drift_speed_evalutation, hpo, overfitting are used to create addtional results and to improve the hyperparameters

***

To run the notebook create the folder 'Thesis/datasets_streaming/...' and upload all datasets, which can be found at <https://onedrive.live.com/?cid=6e1780e1737b1df0&id=6E1780E1737B1DF0%21203&authkey=%21ANx05ZJlM%2Dx88dM>

***


In the default setting we also lead the optimized hyperparameters for all approaches. This can also be adjusted in the config ("use_best_hpo_result").

By default, the GPU is used for the calculations of GDT. This can be changed in the config ("use_gpu"). If you receive a memory error for GPU computations, this is most likely due to the parallelization of the experiments on a single GPU. Please try to reduce the number of parallel computations ("n_jobs") or run on CPU ("use_gpu").

