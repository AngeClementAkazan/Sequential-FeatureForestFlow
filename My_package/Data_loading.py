from sklearn.datasets import load_iris,load_wine
import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
import os
import copy
import wget
import zipfile
import warnings
warnings.simplefilter('ignore')
def data_loader(dt_name):

    if dt_name=="iris":
        my_data=load_iris()
        X,y=my_data["data"],my_data["target"]
        mask_cat=[False]*4+[True]

    elif dt_name=="wine":
        my_data=load_wine()
        X,y=my_data["data"],my_data["target"]
        mask_cat=[False for i in range(X.shape[1])] +[True]
    elif dt_name=="congress":
        # fetch dataset 
        my_data=fetch_congress() 
        X=my_data['data'].astype(int)
        y=my_data['target'].astype(int)

        mask_cat=[True, True, True, True, True, True, True, True ,True, True, True,True, True,True, True,True,True]
    elif dt_name=="heart_disease":
        my_data = fetch_ucirepo(id=45) 
        # data (as pandas dataframes) 
        X = my_data.data.features 
        y = my_data.data.targets 
        Xy=pd.concat([X, y], axis=1)
        Xyy=Xy.dropna().copy()
        X=Xyy.values
        y=X[:,-1]
        new_perm = np.random.permutation(X.shape[0])
        np.take(X, new_perm, axis=0, out=X)
        np.take(y, new_perm, axis=0, out=y)
        mask_cat = [False, True, True, False, False,True, True, False,True, False, True, False,True,False]
        return X,y,my_data,mask_cat
    elif dt_name=="tic-tac-toe":
        my_data=fetch_tictactoe()
        X=my_data['data']
        y=my_data['target']
        mask_cat=[True, True, True ,True, True, True,True, True,True,True]   
    elif dt_name== "breast_cancer":
        my_data=fetch_breast_cancer_diagnostic()
        X=my_data['data']
        y=my_data['target']
        mask_cat=[False]*30+[True]
    elif dt_name== "ionosphere":
        my_data=fetch_ionosphere()
        X=my_data['data']
        y=my_data['target']
        mask_cat=[True]+[False]*32+[True]
    else:
        raise Exception('Download your data')
    new_perm = np.random.permutation(X.shape[0])
    np.take(X, new_perm, axis=0, out=X)
    np.take(y, new_perm, axis=0, out=y)
    X=np.concatenate((X, y.reshape(-1,1)), axis=1)
    X, y = copy.deepcopy(X), copy.deepcopy(y)
    return X,y,my_data,mask_cat    



def fetch_tictactoe():
    dataset_dir = '/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/tictactoe'
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
        
    url = 'https://archive.ics.uci.edu/static/public/101/tic+tac+toe+endgame.zip'
    wget.download(url, out=dataset_dir)

    zip_path = os.path.join(dataset_dir, 'tic+tac+toe+endgame.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    data_path = os.path.join(dataset_dir, 'tic-tac-toe.data')
    df = pd.read_csv(data_path, delimiter=',', header=None)
   
    dataset = {}
    dataset['data'] = np.zeros(df.values[:, :-1].shape)
    
    for i in range(dataset['data'].shape[1]):
        dataset['data'][:, i] = pd.factorize(df.values[:, i])[0].astype(int)
    
    dataset['target'] = pd.factorize(df.values[:, -1])[0].astype(int)

    return dataset
def fetch_congress():
    dataset_dir = '/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/Congress'
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    url = 'https://archive.ics.uci.edu/static/public/105/congressional+voting+records.zip'
    wget.download(url, out=dataset_dir)
    
    zip_path = os.path.join(dataset_dir, 'congressional+voting+records.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    data_path = os.path.join(dataset_dir, 'house-votes-84.data')
    df = pd.read_csv(data_path, delimiter=',', header=None)
    
    dataset = {}
    dataset['data'] = np.zeros(df.values[:,1:].shape)
    
    for i in range(dataset['data'].shape[1]):
        dataset['data'][:, i] = pd.factorize(df.values[:, i])[0].astype(int)
    
    dataset['target'] = pd.factorize(df.values[:, 0])[0].astype(int)

    return dataset

def fetch_breast_cancer_diagnostic():
    if not os.path.isdir('/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/fetch_breast_cancer'):
        os.mkdir('/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/fetch_breast_cancer')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out='/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/fetch_breast_cancer')

    with open('/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/fetch_breast_cancer/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] = pd.factorize(df.values[:, 1])[0] # str to numeric

    return Xy

def fetch_ionosphere():
    if not os.path.isdir('/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/ionosphere'):
        os.mkdir('/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out='/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/ionosphere')

    with open('/Users/ange-clementakazan/Documents/DIFFUSION_MODELS/Data_set/ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = np.concatenate((df.values[:, 0:1].astype('float'), df.values[:, 2:-1].astype('float')), axis=1) # removing the secon variable which is always 0
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy
def fetch_random_data(perct_cat,num_samples,num_features):
    # Set parameters
    cont_var_n=  num_features-num_features*perct_cat
    cat_var_n=  num_features*perct_cat
    #Build the continuous block
    # Mean vector
    mean = np.zeros(int( cont_var_n))
    # Covariance matrix (identity matrix)
    cov_matrix = np.eye(num_features)
    # Generate continous data from multivariate Gaussian distribution
    cont_data = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)
    

    # # Convert to pandas DataFrame
    # columns = [f'feature_{i}' for i in range(num_features)]
    # df = pd.DataFrame(data, columns=columns)


# num_samples = 2000  # Total number of samples
# num_features = 15   # Total number of features
# cont_var_n= 15
# perct_cat=0.8

# # Mean vector
# mean =int(num_features*perct_cat)
# print(mean)