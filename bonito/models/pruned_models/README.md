# Pruned models

We include a subset of our pruned models generated as part of the project in the case that our project is used as a foundation for future work.
To avoid having to retrain models using our methodology, we provide our five most pruned models derived using our global unstructured pruning algorithm. 

Only models without additional pruning parameterisation are included. These are stored using sparse tensors, and their weight files are suffixed by `_sparse.tar`.

Models with pruning parameterisation would contain the additional bit masks used during pruning as a parameter, increasing the model size. However, this means further pruning can be performed on them. Due to the constraints on filesize on GitHub, these are not included.
