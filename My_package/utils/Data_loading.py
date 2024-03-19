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

class data_loading:
    def __init__(self,dt_name, perct_cat=None ,num_samples=None,num_features=None):
        self.perct_cat=perct_cat
        self.num_samples=num_samples
        self.num_features=num_features
        self.dt_name=dt_name
    def data_loader(self):
        if self.dt_name=="iris":
            my_data=load_iris()
            X,y=my_data["data"],my_data["target"]
            mask_cat=[False]*4+[True]

        elif self.dt_name=="wine":
            my_data=load_wine()
            X,y=my_data["data"],my_data["target"]
            mask_cat=[False for i in range(X.shape[1])] +[True]
        elif self.dt_name=="congress":
            # fetch dataset 
            my_data=fetch_congress() 
            X=my_data['data'].astype(int)
            y=my_data['target'].astype(int)

            mask_cat=[True, True, True, True, True, True, True, True ,True, True, True,True, True,True, True,True,True]
        elif self.dt_name=="heart_disease":
            my_data = fetch_ucirepo(id=45) 
            # data (as pandas dataframes) 
            X = my_data.data.features 
            y = my_data.data.targets 
            Xy=pd.concat([X, y], axis=1)
            Xy_c=Xy.dropna().values
            X,y=Xy_c[:,:-1],Xy_c[:,-1]
            mask_cat = [False, True, True, False, False,True, True, False,True, False, True, False,True,True]
        elif self.dt_name=="tic-tac-toe":
            my_data=fetch_tictactoe()
            X=my_data['data']
            y=my_data['target']
            mask_cat=[True, True, True ,True, True, True,True, True,True,True]   
        elif self.dt_name== "breast_cancer":
            my_data=fetch_breast_cancer_diagnostic()
            X=my_data['data']
            y=my_data['target']
            mask_cat=[False]*30+[True]
        elif self.dt_name== "ionosphere":
            my_data=fetch_ionosphere()
            X=my_data['data']
            y=my_data['target']
            mask_cat=[True]+[False]*32+[True]
        else:
            raise Exception('Download your data')
        new_perm = np.random.permutation(X.shape[0])
        np.take(X, new_perm, axis=0, out=X)
        np.take(y, new_perm, axis=0, out=y)
        X_y=np.concatenate((X, y.reshape(-1,1)), axis=1)
        Xy, y = copy.deepcopy(X_y), copy.deepcopy(y)
        return Xy,y,my_data,mask_cat    
    def random_data_loader(self):
        # Set parameters
        cat_var_nb=  int( self.num_features*self.perct_cat)
        cont_var_nb=  self.num_features-cat_var_nb
        cat_data= np.random.binomial(n=1, p=0.5, size=(self.num_samples,cat_var_nb))
        cat_data[cat_data==0]=-1
        if cont_var_nb!=0:
            #Build the continuous block
            mean = np.zeros(cont_var_nb)
            cov_matrix = np.eye(cont_var_nb)
            #Generate continous data from multivariate Gaussian distribution
            cont_data = np.random.multivariate_normal(mean, cov_matrix, size=self.num_samples)
            data_= np.concatenate((cat_data,cont_data), axis=1)
        else:
            data_= cat_data
        col_num= data_.shape[1]
        y=None
        if self.dt_name== "Maj_sum>0":
            y= np.array([  1 if np.sum(row)>0 else 0 for row in data_]  ).reshape(-1,1)
        elif self.dt_name== "Maj_vote>0":
            y= np.array([  1 if sum(row>0)>= col_num//2 else 0 for row in data_]  ).reshape(-1,1)
        elif self.dt_name== "XORofAllSign":
            y=np.array([ XOR_of_signs(row)  for row in data_] ).reshape(-1,1)
        else:
            raise Exception("Choose your own experiment")
        Xy= copy.deepcopy(np.concatenate((data_,y), axis=1))
        Xy[Xy == -1] = 0
        mask=[True]*cat_var_nb+[False]*cont_var_nb+[True]
        return Xy,y,data_,mask

def XOR_of_signs(ls):
    result = 0
    for num in ls:
        # Get the sign of the number (1 for positive, 0 for negative)
        sign = 1 if num >= 0 else 0
        # Perform XOR operation with the result
        result ^= sign
    return result

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
        Xy['data'] = np.concatenate((df.values[:, 0:1].astype('float'), df.values[:, 2:-1].astype('float')), axis=1) # removing the second variable which is always 0
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric
    return Xy

