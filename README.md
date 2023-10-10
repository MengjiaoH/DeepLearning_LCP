### Install 
1. PyTorch https://pytorch.org/get-started/locally/
2. tqdm `pip install tqdm`
3. scikit-learn `pip install scikit-learn`
   

### Training Data Sets

The training data sets are in ".npy" format. Each data sample is a vector of size 16.

1. Mean values: index 0 to index 3
2. Covariance values: index 4 to index 13
3. Iso-value: index 14
4. Level crossing probability: index 15 

To change the method of loading data (e.g. use a different data format), edit the init function if the dataloader.py.

### Training 

The parameter of the *batch_size*, *nepochs*, *lr*, *data_dir*, *k_folds* and *savefolder* can be edited in train.py. 

After setting the parameters, run the training process as follows: 

`python train.py`

