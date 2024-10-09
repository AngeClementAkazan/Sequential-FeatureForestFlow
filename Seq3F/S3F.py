import math
import numpy as np
import copy
import xgboost as xgb
from functools import partial
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from lightgbm import LGBMRegressor,LGBMClassifier
# from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from joblib import delayed, Parallel
from scipy.special import softmax
import random


# Class for the SF3 framework
# Categorical features should be numerical (rather than strings), make sure to use x = pd.factorize(x)[0] to make them as such
# Make sure to specific which features are categorical and which are integers
# If you have categorical variable whose label does not respect {0,..,N} for instance {1,...,N}, try to label encod the data first to make the label starting from zero to the Nth label

class feature_forest_flow():
  def __init__(self, 
               X, # Must be a Numpy dataset and 
               label_y=None, # # must be a categorical/binary variable| When None then X shopuld be the entire data set
               cat_y=True, # Binary variable indicating whether or not the output is categorical
               n_t=51, # number of noise level
               model='xgboost', # xgboost, random_forest, lgbm, catboost,
               solver_type='Rg4', # solver type: argument (Euler or Rg4)
               model_type='HS3F', #HS3F for heterogenuous  and  CS3F for regressor only,
               one_hot_encoding=False,
               duplicate_K=100, # number of different noise sample per real data sample
              #  bin_indexes=[], # vector which indicates which column is binary 
               cat_indexes=[], #Vector indicating which column is categorical/binary (the classes must be under the form [0,1,...,J]) if not, use label encoding before using your data
               int_indexes=[], # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
               max_depth = 7, n_estimators = 100, eta=0.3,   learning_rate=0.3, # xgboost hyperparameters
               tree_method='hist', reg_alpha=0.0, reg_lambda = 0.0, subsample=1.0, # xgboost hyperparameters
               num_leaves=31, # lgbm hyperparameters
               true_min_max_values=None, # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
               gpu_hist=False, # using GPU or not with xgboost
               n_z=10, # number of noise to use in zero-shot classification
               eps=0, 
               n_jobs=-1, # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
               n_batch=0, # If >0 use the data iterator with the specified number of batches
               ngen=5, # The number generated batches, it will help us to limit the multinomial sampling in case the first variable of the data is categorical based on the class frequencies of this variable 
               seed=666,
               label_format="XgbClassifier", #Make sure the label are formatted according to the Xgboost classifier format [0,...,Max(Class)]
               arg1={},arg2={}
               ): # you can pass extra parameter for xgboost using arg1 for Classifier and  arg2 for Xgb.Regressor

    assert isinstance(X, np.ndarray), "Input dataset must be a Numpy array"
    assert len(X.shape)==2, "Input dataset must have two dimensions [n,p]"
    assert "All the category must be in the form [0,...,N], use label encoding if it is not the case"
    np.random.seed(seed)
    self.ngen=ngen 
    self.one_hot_encoding=one_hot_encoding
    self.n_t = n_t 
    self.duplicate_K = duplicate_K
    self.model = model
    self.solver_type=solver_type
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.seed = seed
    self.num_leaves = num_leaves
    self.eta = eta,
    self.learning_rate=  learning_rate
    self.gpu_hist = gpu_hist
    self.n_jobs = n_jobs
    self.tree_method = tree_method
    self.reg_lambda = reg_lambda
    self.reg_alpha = reg_alpha
    self.subsample = subsample
    self.n_z = n_z
    self.eps = eps
    self.n_batch = n_batch
    self.model_type=model_type
    self.t_levels = np.linspace(self.eps, 1, num=self.n_t)
    self.arg1 = arg1
    self.arg2 = arg2
    self.ls=[]
    # try:
    #   int_indexes = int_indexes + bin_indexes # since we round those, we do not need to dummy-code the binary variables
    # except:
    #    int_indexes=[]
    # Categorical input encompasses binary input
    # try:
    #    cat_indexes= cat_indexes +bin_indexes
    # except:
    #    cat_indexes=cat_indexes
    self.cat_y = cat_y
    self.int_indexes = int_indexes
    self.cat_indexes=cat_indexes
    # self.bin_indexes=bin_indexes
    
    # Construct a categorical mask for the variable (True if categorical and False otherwise)
    if label_y is not None:
       X_dim_size=X.shape[1]+1
    else:
       X_dim_size=X.shape[1]
    if self.cat_indexes is not None:
       if  self.cat_y== True :        
          mask_cat_bf= [i in self.cat_indexes  for i in range(X_dim_size-1)]+[True]  #Correct
       else:
          mask_cat_bf= [i in self.cat_indexes  for i in range(X_dim_size-1)]+[False]  #Correct
    else:
       if  self.cat_y== True:      
          mask_cat_bf=[False]*(X_dim_size-1) +[True]
       else:
          mask_cat_bf=[False]*X_dim_size
    #Jesus
     # Remove all missing values and shuffle data
    obs_to_remove = np.isnan(X).any(axis=1)
    X = X[~obs_to_remove]
    # print(np.unique(X[:,-1]))
    if label_y is not None:
      #  X=X[:,:-1]
       mask_cat_bf= mask_cat_bf[:-1]
       label_y = label_y[~obs_to_remove]     
    if true_min_max_values is not None:
        self.X_min = true_min_max_values[0]
        self.X_max = true_min_max_values[1]
    else:
        self.X_min = np.nanmin(X, axis=0, keepdims=1)
        self.X_max = np.nanmax(X, axis=0, keepdims=1)

    #Set the label conditionning conditions
    self.label_y = label_y
    self.mask_cat_bf= mask_cat_bf
    self.cat_indexes_=[i for i in range(len(self.mask_cat_bf)) if self.mask_cat_bf[i] ] # Index of categorical variables before on hot encoding
    #Check if the categories are under the form [0,..., Max(classes)] it is important for the Xgboost CLassifier
    # check_categorical_columns(X, self.cat_indexes_)
    #Number of  categorical variables
    num_cat_index=len( self.cat_indexes_)
    self.num_cat_index=num_cat_index      
    mask_cat=copy.deepcopy(self.mask_cat_bf) 

    #min-max normalization, this does not apply to the categorical data because they will be handled by a classifier
    self.scaler = MinMaxScaler(feature_range=(-1, 1))
    if self.num_cat_index  < len(self.mask_cat_bf):
        X[:,~np.array(self.mask_cat_bf)]=self.scaler.fit_transform(X[:,~np.array(self.mask_cat_bf)])

    #if there is categorical variable and and the variable one_hot_encoding is set to True
    if self.num_cat_index > 0 and self.one_hot_encoding: 
        X, self.X_names_before, self.X_names_after,mask_cat= self.dummify(X) # dummy-coding for categorical variables 
    self.X=X
    self.b , self.c = self.X.shape
    self.row_number=self.ngen*self.b
    self.mask_cat=mask_cat

    if model == 'random_forest' and np.sum(np.isnan(X)) > 0:
      raise  Exception('The dataset must not contain missing data in order to use model=random_forest')
    X1=copy.deepcopy(self.X)
    label_y_=self.label_y
    if self.n_batch == 0: 
      if duplicate_K >= 1: # we duplicate the data multiple times, so that X0 is k times bigger so we have more room to learn
        if self.b >= 10000:
          print("Dataset is too large, we recommend switch to mini-batch-style training by setting n_batch > 1")
        X1 = np.tile(X1, (duplicate_K, 1))
        label_y_=np.tile(self.label_y, duplicate_K)
    row_X1,_=X1.shape
    self.X1=X1
    # print(self.X1.shape,duplicate_K)
    #Set the label conditionning conditions
    if self.label_y is not None:
      assert np.sum(np.isnan(self.label_y)) == 0 # cannot have missing values in the label (just make a special categorical for nan if you need)
      self.y_uniques, self.y_probs = np.unique(label_y_, return_counts=True)
      self.y_probs = self.y_probs/np.sum(self.y_probs)
      self.mask_y = {} # mask for which observations has a specific value of y
      for i in range(len(self.y_uniques)):
        self.mask_y[self.y_uniques[i]] = np.zeros(row_X1, dtype=bool)
        self.mask_y[self.y_uniques[i]][label_y_ == self.y_uniques[i]] = True
    else: # assuming a single unique label 0
      self.y_probs = np.array([1.0])
      self.y_uniques = np.array([0])
      self.mask_y = {} # mask for which observations has a specific value of y
      self.mask_y[0] = np.ones(X1.shape[0], dtype=bool)
  
  # Define Xgb.Regressor model to learn velocity flow for continuous variables and Xgb.Classifier model to learn discrete variables
  def train_cont_cat(self, X_train, y_train, k):  #The training models  
      y_no_miss = ~np.isnan(y_train.ravel())
      y_train=y_train[y_no_miss]
      X_train=X_train[y_no_miss, :]     
      if self.mask_cat[k] and self.model_type== "HS3F":
          if self.model == 'random_forest':
              out = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.seed)
          elif self.model == 'xgboost':
                # y_train.astype("category")
                objective = 'binary:logistic' 
                out = xgb.XGBClassifier(n_estimators=self.n_estimators+15, objective=objective,learning_rate=0.1,max_depth=self.max_depth, reg_lambda=self.reg_lambda, 
                                        reg_alpha=self.reg_alpha, subsample=self.subsample, seed=self.seed,tree_method=self.tree_method,n_jobs=self.n_jobs,  enable_categorical=True,
                                          device='cuda' if self.gpu_hist else 'cpu', **self.arg1)
          else:
              raise Exception("model value does not exist")               
          out.fit(X_train, y_train)
          return out
      elif not self.mask_cat[k] or self.model_type== "CS3F":
          if self.model == 'random_forest':
              out = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.seed)
          elif self.model == 'xgboost':
              out = xgb.XGBRegressor(n_estimators=self.n_estimators, objective='reg:squarederror', eta=0.3, max_depth=self.max_depth, 
              reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, subsample=self.subsample, random_state=self.seed,seed=self.seed, tree_method=self.tree_method,n_jobs=self.n_jobs, 
              device='cuda' if self.gpu_hist else 'cpu', **self.arg2)
          else:
              raise Exception("model value does not exists")    
          out.fit(X_train, y_train)
          return out
      else:
         raise Exception(" Choose model_type or check the variable mask")
  
  # MiniBatch training using Dmatrix
  def train_iterator(self, X1_splitted,n_t, dim,t, i, j): 
      np.random.seed(self.seed)
      it = IterForDMatrix(X1_splitted,self.mask_cat,n_t, self.get_xt_y,self.make_mat,t,i,dim,self.model_type, n_batch=self.n_batch, n_epochs=self.duplicate_K)
      data_iterator = xgb.QuantileDMatrix(it)
      if self.mask_cat[dim] and self.model_type=="HS3F":
            objective="multi:softprob"
            num_class=len(np.unique(self.X1[:,dim], return_counts=False)) 
            lr=0.1
      else:
          objective='reg:squarederror' 
          num_class= None
          lr=self.learning_rate
      xgb_dict = {'objective':objective,'max_depth': self.max_depth,"learning_rate":lr,
            "reg_lambda": self.reg_lambda, 'reg_alpha': self.reg_alpha, "subsample": self.subsample, "seed": self.seed, 
              "tree_method": self.tree_method, 'device': 'cuda' if self.gpu_hist else 'cpu', "num_class":num_class,
              "device": "cuda" if self.gpu_hist else 'cpu'}
      if self.mask_cat[dim] and  self.model_type=="HS3F": # Additional arguments for Xgb.Classifier
          if len(self.arg2)>0:
              for myarg in self.arg2:              
                      xgb_dict[myarg] = self.arg2[myarg]
      else:
          if len(self.arg1)>0:   # Additional arguments for Xgb.Regressors
              for myarg in self.arg1:              
                  xgb_dict[myarg] = self.arg1[myarg]
      out = xgb.train(xgb_dict, data_iterator, num_boost_round=self.n_estimators)
      return out
  
  #Multinomial function that is used in case the first variable of the data is categorical as we do not use diffusion steps for dicrete data
  def samp_mult(self,X):
      x0_uniques, x0_probs = np.unique(X.reshape(-1,1),return_counts=True)
      x0_probs = x0_probs/np.sum(x0_probs)
      x0_2get=x0_uniques[np.argmax(np.random.multinomial(1, x0_probs, size=(self.row_number,)), axis=1)]
      return x0_2get
  #This define the forward process (Condition flow and velocity vector gathering for time step t and all feature )
  def forward_process(self,X,sigma=0.0): 
      b, c =  X.shape       
      X_train = np.zeros((c, self.n_t, b,1))              # [c,n_t, b*100, 1]  # Will contain the interpolation between x0 and x1 (xt)   
      y_train = np.zeros((c, self.n_t,b, 1))               # [c,n_t, b*100, 1]  # Will contain the output to predict (ut).reshape(-1,1 
      X0 = np.random.normal(size=(b,c))  
      eps=np.random.randn(*X.shape )   
      for j in range(c):                      
          for i in range(self.n_t):               
              t_ = np.ones( X.shape[0])*self.t_levels[i]
              t=t_.reshape(-1,1) # current t
              xt, ut =  t*X[:,j].reshape(-1,1)+ (1-t)*X0[:,j].reshape(-1,1), X[:,j].reshape(-1,1)-X0[:,j].reshape(-1,1)  # Fill the containers previously initialized with xt and ut
              X_train[j][i][:], y_train[j][i][:] = xt+sigma*eps[:,j].reshape(-1,1), ut.reshape(-1,1)
      return X_train, y_train,X
 #Get conditional flow and velocity vector for a given t, this helpful for the training by batch
  def get_xt_y(self,X,k,t,i): 
    b,_=  X.shape
    X0 = np.random.normal(size=(b,1))        
    xt, ut =  t*X[:,k].reshape(-1,1)+ (1-t)*X0, X[:,k].reshape(-1,1)-X0
    return xt, ut 
  # Make Datasets of interpolation (gather features in order to create training input for Xgb.Regressor and Xgb.Classifier)
  def make_mat(self,Mat,k,x_chunk=None):  
      if self.mask_cat[k] and self.model_type=="HS3F":
          A=()
          if k==1:
              X_train=Mat[:,k-1].reshape(-1,1)
          elif k>1:
            for h in range(k):
                A+=(Mat[:,h].reshape(-1,1),)  # A is initialized with the first variable  and will contain all the variable precedenting the one we want to predict using our classifier
            X_train=np.concatenate(A,axis=1)
          else:
              print("k=0 is not desirable")
          return X_train
      else: 
          if k==0:
            return x_chunk
          else:        
            A=(x_chunk,)
            for h in range(k):
                A+=(Mat[:,h].reshape(-1,1),)
            X_train=np.concatenate(A,axis=1)             
            return X_train  
  # Training Process 
  def training_(self,X_):  
    b, c =  X_.shape
    # Initialize the containers
    results_cont = [] # Will contain the trained models for continuous data
    results_cat = []   # Will contain the trained models for categorical data
    if self.model_type== "HS3F":
        cont_list=[i for i in range(len(self.mask_cat)) if not self.mask_cat[i]]    #Get all the indices for continuous variable
        regr_ = [[[None  for i in range(self.n_t)] for j in self.y_uniques ]  for p in range(len(cont_list)) ]   #Initialize a container that will receive the Xgboost Regressors build to predict noisy outputs  
    elif self.model_type=="CS3F":
        regr_=[[[None  for i in range(self.n_t)] for j in self.y_uniques] for k in range(c)]        
    else:
        raise Exception ( " Choose the right value for the  model_type argument")   
    f, g,X = self.forward_process(X_) # Get f and g are respectively conditional flow and  velocity vector for t and and k
    # In case the user chose the training by batch
    if self.n_batch > 0: #no need to duplicate
      rows_per_batch = b//self.n_batch
      batches = [rows_per_batch*i  for i in range(1,self.n_batch)]  # rpb=5, batch=5, b=25 ls1: [5,10,15,20]+[25-5*4]+ [self.b - rows_per_batch*(self.n_batch-1)]
      X1_splitted = {}
      for i in self.y_uniques:
        X1_splitted[i] = np.split(X[self.mask_y[i], :], batches, axis=0)
    if self.n_jobs==1:      
        if self.model_type=="HS3F":  #To choose mixture of model
              if self.n_batch>0: 
                  for k in range(c):
                      if self.mask_cat[k]:    
                          if k==0: #if the first variable is categorical, we use multinomial sampling(epecially when the whole data is discrete)                     
                              results_cat.append(self.samp_mult(X[:,k]) ) 
                          else:      
                              for j in range(len(self.y_uniques)):
                                  i=0 # No diffudion steps
                                  #Gather Xgb.Classifier model trained per batch on previous features  
                                  results_cat.append(self.train_iterator( X1_splitted[j], self.n_t, k, self.t_levels[i], i, j))
                      else: 
                          for j in range(len(self.y_uniques)):
                              for i in range(self.n_t):  
                                  #Gather Xgb.Regressor model trained per batch on previous features      
                                  results_cont.append(self.train_iterator(X1_splitted[j],self.n_t,k,self.t_levels[i], i, j)) 
              else:
                  for k in range(c):
                      if self.mask_cat[k]:  #if the categorical mask is true then we do the following operations    
                          if k==0:                           
                              results_cat.append(self.samp_mult(X[:,k]) ) 
                          else:
                                                                            
                              for j in range(len(self.y_uniques)):                                      
                                  Yy=X[self.mask_y[j],k]
                                  X_train=self.make_mat(X[self.mask_y[j]],k)
                                  result= self.train_cont_cat(X_train,Yy,k)
                                  results_cat.append(result)
                      else:   #if the categorical mask is False then we do the following operations  
                          for j in range(len(self.y_uniques)): 
                              for i in range(self.n_t):          
                                  X_train_chunk,y_train_chunk= f[k][i][self.mask_y[j]], g[k][i][self.mask_y[j]]
                                  X_train=self.make_mat(X[self.mask_y[j]],k,X_train_chunk)                                                     
                                  result = self.train_cont_cat(X_train,y_train_chunk,k)
                                  results_cont.append(result)
              current_i_cont = 0              
              for kk in range(len(cont_list)):
                      for j in range(len(self.y_uniques)):
                          for i in range(self.n_t):
                                  regr_[kk][j][i] = results_cont[current_i_cont]
                                  current_i_cont += 1
              return regr_,results_cat 
        elif self.model_type== "CS3F":  #To choose only model for continuous settings
            if self.n_batch>0: 
                for k in range(c):
                    for j in range(len(self.y_uniques)):
                        for i in range(self.n_t):                                                                                         
                            results_cont.append(self.train_iterator(X1_splitted[j],self.n_t,k,self.t_levels[i], i, j))
            else:
                for k in range(c):
                    for j in range(len(self.y_uniques)): 
                        for i in range(self.n_t):          
                            X_train_chunk,y_train_chunk= f[k][i][self.mask_y[j]], g[k][i][self.mask_y[j]]
                            X_train=self.make_mat(X[self.mask_y[j]],k,X_train_chunk)                                                     
                            result = self.train_cont_cat(X_train,y_train_chunk,k)
                            results_cont.append(result)
            current_i_cont = 0
            for kk in range(c):
                for j in range(len(self.y_uniques)):
                    for i in range(self.n_t):
                        regr_[kk][j][i] = results_cont[current_i_cont]
                        current_i_cont += 1
            return regr_,results_cat
        else:
            raise Exception ( " Choose the right value for the  model_type argument")              
    else:  # More than 1 job (paralelle computing)               
      if self.model_type== "HS3F":
          if self.mask_cat[0]:  #If the categorical mask is true then we do the following operations
              results_cat.append(self.samp_mult(X[:,0]) )
          if self.n_batch > 0: 
              results_cat+=Parallel(n_jobs=self.n_jobs)(delayed(self.train_iterator)(X1_splitted[j], self.n_t, k,self.t_levels[0], 0, j) for k in range(1,c) if self.mask_cat[k]  for j in self.y_uniques  )                    
              results_cont+=Parallel(n_jobs=self.n_jobs)(delayed(self.train_iterator)( X1_splitted[j],self.n_t, k,self.t_levels[l],
              l,j) for k in range(c) if not self.mask_cat[k]  for j in self.y_uniques for l in range(self.n_t)  ) 
          else:
              results_cat+=Parallel(n_jobs=self.n_jobs)(delayed(self.train_cont_cat)(self.make_mat(X[self.mask_y[j]] ,k),X[self.mask_y[j],k],k)
              for k in range(1,c) if self.mask_cat[k]  for j in self.y_uniques 
              )
              results_cont+=Parallel(n_jobs=self.n_jobs)(delayed(self.train_cont_cat)
              (self.make_mat(X[self.mask_y[j]],k,f[k][i][self.mask_y[j]]),g[k][i][self.mask_y[j]],k) 
              for k in range(c) if not self.mask_cat[k]  for j in self.y_uniques   for i in range(self.n_t) 
              )
          current_i_cont = 0
          for kk in range(len(cont_list)):
            for j in range(len(self.y_uniques)):
                for i in range(self.n_t):
                    regr_[kk][j][i] = results_cont[current_i_cont]
                    current_i_cont += 1
          
          return regr_,results_cat
      elif self.model_type== "CS3F":
          if self.n_batch > 0:
              results_cont+=Parallel(n_jobs=self.n_jobs)(delayed(self.train_iterator)( X1_splitted[j],self.n_t, k,self.t_levels[l],
              l,j) for k in range(c)  for j in self.y_uniques  for l in range(self.n_t)  ) 
          else:
              results_cont+=Parallel(n_jobs=self.n_jobs)(delayed(self.train_cont_cat)
              (self.make_mat(X[self.mask_y[j]],k,f[k][i][self.mask_y[j]]),g[k][i][self.mask_y[j]],k) 
              for k in range(c)   for j in self.y_uniques  for i in range(self.n_t) 
              )        
          current_i_cont = 0
          for kk in range(c):
              for j in range(len(self.y_uniques)):
                  for i in range(self.n_t):
                      regr_[kk][j][i] = results_cont[current_i_cont]
                      current_i_cont += 1               
          return regr_,results_cat
      else:
              raise Exception(" Choose the correct argument model_type")

  def dummify(self, X):
    df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
    df_names_before = df.columns
    for i in self.cat_indexes_:
      df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=True)
    df_names_after = df.columns
    cat_mask_indexes = []  #List to store categorical column indexes
    for j in df_names_after:
        if "_" in j:  #Check if the last added column has "_"
            cat_mask_indexes.append(True)
        else:
            cat_mask_indexes.append(False)
    df = df.to_numpy()
    return df, df_names_before, df_names_after,cat_mask_indexes

  def unscale(self, X):
    if self.scaler is not None: # unscale the min-max normalization
      X = self.scaler.inverse_transform(X)
    return X
  
  # Rounding for the categorical variables which are dummy-coded and then remove dummy-coding
  def clean_onehot_data(self, X):
    if len(self.cat_indexes_) > 0: # ex: [5, 3] and X_names_after [gender_a gender_b cartype_a cartype_b cartype_c]
      X_names_after = copy.deepcopy(self.X_names_after.to_numpy())
      prefixes = [x.split('_')[0] for x in self.X_names_after if '_' in x] # for all categorical variables, we have prefix ex: ['gender', 'gender']
      unique_prefixes = np.unique(prefixes) # uniques prefixes
      for i in range(len(unique_prefixes)):
        cat_vars_indexes = [unique_prefixes[i] + '_' in my_name for my_name in self.X_names_after]
        cat_vars_indexes = np.where(cat_vars_indexes)[0] # actual indexes
        cat_vars = X[:, cat_vars_indexes] # [b, c_cat]
        # dummy variable, so third category is true if all dummies are 0
        cat_vars = np.concatenate((np.ones((cat_vars.shape[0], 1))*0.5,cat_vars), axis=1)
        # argmax of -1, -1, 0 is 0; so as long as they are below 0 we choose the implicit-final class
        max_index = np.argmax(cat_vars, axis=1) # argmax across all the one-hot features (most likely category)
        X[:, cat_vars_indexes[0]] = max_index
        X_names_after[cat_vars_indexes[0]] = unique_prefixes[i] # gender_a -> gender
      df = pd.DataFrame(X, columns = X_names_after) # to Pandas
      df = df[self.X_names_before] # remove all gender_b, gender_c and put everything in the right order
      X = df.to_numpy()
    return X

  # Unscale and clip to prevent going beyond min-max and also round of the integers
  def clip_extremes(self, X):
    if self.int_indexes is not None:
      for i in self.int_indexes :
        X[:,i] = np.round(X[:,i], decimals=0)
    for i in range(len(self.mask_cat_bf)):
       if self.mask_cat_bf[i] and self.model_type=="CS3F":
         X[:,i] = np.round(X[:,i], decimals=0)
    small = (X < self.X_min).astype(float)
    X = small*self.X_min + (1-small)*X
    big = (X > self.X_max).astype(float)
    X = big*self.X_max + (1-big)*X
    return X 
  
 # Building the continuous feature generation model 
  def my_model_cont(self,tr_container,noise,j,label,t,k,cont_count,dmat,mask_y,x_k, x_prev):
      row_noise=noise.shape[0]
      out = np.zeros((row_noise,self.c)) # [b, c]
      i = int(round(t*(self.n_t-1)))
      if x_prev is None:                
          x = x_k    #If no previous variable (x_prev==None), x is the noisy input data used to generate the first variable of the data
      else:         #x receives the previous variable having been 
          x = np.concatenate((x_k, x_prev), axis=1) # We respect the training structure for continuous variable that is: the model reveives (X_noise, Variable1,...,Variable k-1) to predict Variable k    
      x_=x[mask_y[label]]
      if dmat:
        X=xgb.DMatrix(data=x_)
        out[mask_y[label], k] = tr_container[0][cont_count][j][i].predict(X)
        return out
      else:          
        out[mask_y[label], k] = tr_container[0][cont_count][j][i].predict(x_)   
        return out
      
 # Building the categorical feature generation model 
  def my_model_cat(self,tr_container,noise,j,label,k,cat_count,dmat,mask_y, x_prev):
      row_noise=noise.shape[0]
      out = np.zeros((row_noise,self.c)) # [b, c]
      if x_prev is None and k==0:
          out[:, k] = tr_container[1][cat_count][: row_noise] # multinomial sampling for k=0
          return out
      else:
          x_=x_prev[mask_y[label]]
          if dmat:
            x_prev_=xgb.DMatrix(data=x_)
            x_pred=tr_container[1][cat_count].predict(x_prev_)
            row,col=x_pred.shape               
            x_fake = np.zeros(row)
            y_categories=np.array([i for i in range(col)])
            for j in range(row):
                x_fake[j] = y_categories[np.argmax(np.random.multinomial(1, x_pred[j], size=1), axis=1)][0] # sample according to probability
            out[mask_y[label], k] =x_fake            
            return out 
          else:
            x_pred=tr_container[1][cat_count].predict_proba(x_)
            row,col=x_pred.shape               
            x_fake = np.zeros(row)
            y_categories=np.array([i for i in range(col)])
            for j in range(row):
                x_fake[j] = y_categories[np.argmax(np.random.multinomial(1, x_pred[j], size=1), axis=1)][0] # sample according to probability
            out[mask_y[label], k] =x_fake
            return out
  # # Sometime the classes does not respect the format [0,...J], in this case we label encode these classes and use the function below to unlabelencode them
  # def label_inverse(self,y_pred):
  #     for i in np.unique(y_pred, return_counts=False):
  #         y_pred[y_pred==i]=i+1
  #     return y_pred
     
  # Euler ODE solver
  def euler_solve(self,tr_container,noise,x_k,label,mask_y,dmat):     
      h = 1 / (self.n_t-1)
      x_prev = None     #Used to store the generated column x_{t-1} to generated column x_t
      A=()     #A is the container that will receive all the generated variables
      #The cat_count and cont_count argument serve to pick the right model respectively from the categorical and continuous list of models
      cat_count=0
      cont_count=0
      row_noise=noise.shape[0]
      if self.model_type == "HS3F":
        for k in range(self.c):
            if self.mask_cat[k]:
                x_k=np.zeros((row_noise,1))
                if k==0:
                  j=label=0
                  x_k=self.my_model_cat(tr_container,noise,j,label,k,cat_count,dmat,mask_y, x_prev=x_prev)[:,k].reshape(-1,1)
                  cat_count+=1
                else:
                  for j, label in enumerate(self.y_uniques):
                    x_k+=self.my_model_cat(tr_container,noise,j,label,k,cat_count,dmat,mask_y, x_prev=x_prev)[:,k].reshape(-1,1)
                    cat_count+=1
                    # # In case the class  format does not match with the xgboost classifier expected format ([0,1,...,J])
                    # if not (np.unique(self.X[:,mask_y[label]], return_counts=False)==np.unique(x_k, return_counts=False)).all():
                    #       x_k=self.label_inverse(x_k) 
            else:
                ##Choose  ===>  x_k=np.random.normal(size=(row_noise,1)) for the default ODE initial condition and ===> x_k=np.random.normal(loc=0.1, scale=1.1,size=(row_noise,1)) for Modified initial condition
                x_k=np.random.normal(size=(row_noise,1))
                for j, label in enumerate(self.y_uniques):
                  t=0           
                  for i in range(self.n_t):  #Loop for numerical solver
                      x_k+= h*self.my_model_cont(tr_container,noise,j,label,t,k,cont_count,dmat,mask_y,x_k, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                      t = t + h
                cont_count+=1
            if x_prev is None:
                x_prev = x_k  #At k=0, xprev get x_0
            else:
                x_prev = np.concatenate((x_prev,x_k), axis=1) # At k!=0, x_prev receive the previous generated value x_0,...,x_{k-1} plus the generated value of x_k that will be the input for the next generation
            A+=(x_k,)
        A=np.concatenate(A,axis=1)
        return A
      elif self.model_type == "CS3F":
        for k in range(self.c):
          ##Choose  ===>  x_k=np.random.normal(size=(row_noise,1)) for the default ODE initial condition and ===> x_k=np.random.normal(loc=0.1, scale=1.1,size=(row_noise,1)) for Modified initial condition
          x_k=np.random.normal(size=(row_noise,1))
          for j, label in enumerate(self.y_uniques):
            t=0           
            for i in range(self.n_t ):  # Loop for numerical solver
                x_k+= h*self.my_model_cont(tr_container,noise,j,label,t,k,cont_count,dmat,mask_y,x_k, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                t = t + h
          cont_count+=1
          if x_prev is None:
              x_prev = x_k  #At k=0, xprev get x_0
          else:
              x_prev = np.concatenate((x_prev,x_k), axis=1) # At k!=0, x_prev receive the previous generated value x_0,...,x_{k-1} plus the generated value of x_k that will be the input for the next generation
          A+=(x_k,)
        A=np.concatenate(A,axis=1)
        return A
      
  #Runge Kutta  4th Order ODE solver
  def Rg4(self,tr_container,noise,x_k,label,mask_y,dmat):     
      h = 1 / (self.n_t -1)
      x_prev = None     #Used to store the generated column x_{t-1} to generated column x_t
      A=()     # A is the container that will receive all the generated variables
      ## The cat_count and cont_count argument serve to pick the right model respectively from the categorical and continuous list of models
      cat_count=0
      cont_count=0
      row_noise=noise.shape[0]
      if self.model_type == "HS3F":
        for k in range(self.c):
            if self.mask_cat[k]:
                x_k=np.zeros((row_noise,1)) # should be zeros because we do addition x_k+=self.my_model_cat(..)
                if k==0:
                  j=label=0
                  x_k=self.my_model_cat(tr_container,noise,j,label,k,cat_count,dmat,mask_y, x_prev=x_prev)[:,k].reshape(-1,1)
                  cat_count+=1                 
                else:
                  for j, label in enumerate(self.y_uniques):
                    x_k+=self.my_model_cat(tr_container,noise,j,label,k,cat_count,dmat,mask_y, x_prev=x_prev)[:,k].reshape(-1,1)
                    cat_count+=1               
            else:
                ##Choose  ===>  x_k=np.random.normal(size=(row_noise,1)) for the default ODE initial condition and ===> x_k=np.random.normal(loc=0.1, scale=1.1,size=(row_noise,1)) for Modified initial condition
                x_k=np.random.normal(size=(row_noise,1))
                for j, label in enumerate(self.y_uniques):
                  t=0           
                  for i in range(self.n_t-1):  # Loop for numerical solver
                      k1= h*self.my_model_cont(tr_container,noise,j,label,t,k,cont_count,dmat,mask_y,x_k, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                      k2= h*self.my_model_cont(tr_container,noise,j,label,t+h / 2,k,cont_count,dmat,mask_y,x_k+k1/2, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                      k3= h*self.my_model_cont(tr_container,noise,j,label,t+h / 2,k,cont_count,dmat,mask_y,x_k+k2/2, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                      k4= h*self.my_model_cont(tr_container,noise,j,label,t+h,k,cont_count,dmat,mask_y,x_k+k3, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                      x_k+= (k1 + 2 * k2 + 2 * k3 + k4) / 6
                      t = t + h
                cont_count+=1
            if x_prev is None:
                x_prev = x_k  #At k=0, xprev get x_0
            else:
                x_prev = np.concatenate((x_prev,x_k), axis=1) # At k!=0, x_prev receive the previous generated value x_0,...,x_{k-1} plus the generated value of x_k that will be the input for the next generation
            A+=(x_k,)
        A=np.concatenate(A,axis=1)
        return A
      elif self.model_type == "CS3F":
        for k in range(self.c):
          x_k=np.random.normal(size=(row_noise,1)) #x_k=np.random.normal(loc=0.1, scale=1.1,size=(row_noise,1))is the Affine transformation of x_k
          for j, label in enumerate(self.y_uniques):
            t=0           
            for i in range(self.n_t-1 ):  # Loop for numerical solver
                k1= h*self.my_model_cont(tr_container,noise,j,label,t,k,cont_count,dmat,mask_y,x_k, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                k2= h*self.my_model_cont(tr_container,noise,j,label,t+h / 2,k,cont_count,dmat,mask_y,x_k+k1/2, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                k3= h*self.my_model_cont(tr_container,noise,j,label,t+h / 2,k,cont_count,dmat,mask_y,x_k+k2/2, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                k4= h*self.my_model_cont(tr_container,noise,j,label,t+h,k,cont_count,dmat,mask_y,x_k+k3, x_prev=x_prev)[:,k].reshape(-1,1) # k because we want to return the k th column preddicted by the model
                x_k+= (k1 + 2 * k2 + 2 * k3 + k4) / 6
                t = t + h
          cont_count+=1
          if x_prev is None:
              x_prev = x_k  #At k=0, xprev get x_0
          else:
              x_prev = np.concatenate((x_prev,x_k), axis=1) # At k!=0, x_prev receive the previous generated value x_0,...,x_{k-1} plus the generated value of x_k that will be the input for the next generation
          A+=(x_k,)
        A=np.concatenate(A,axis=1)
        return A  
      
   #The data generation function      
  def generate(self,batch_size):
      x_k=None
      # ODE solve
      dmat=self.n_batch > 0
      train_c=self.training_(self.X1)
      # Generate new data by solving the flow ODE
      noise=np.random.normal(loc=0.1, scale=1.1,size=(self.b if batch_size is None else batch_size, self.c))
      # Generate random labels
      label_y = self.y_uniques[np.argmax(np.random.multinomial(1, self.y_probs, size=noise.shape[0]), axis=1)]
      mask_y = {} # mask for which observations has a specific value of y
      for i in range(len(self.y_uniques)):
        mask_y[self.y_uniques[i]] = np.zeros(noise.shape[0], dtype=bool)
        mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True
      # Euler solver 
      if self.solver_type== "Euler":
        solution = self.euler_solve(train_c,noise,x_k,label_y,mask_y,dmat) 
      # Rg4 Solver
      elif self.solver_type== "Rg4":
        solution = self.Rg4(train_c,noise,x_k,label_y,mask_y,dmat) 
      else:
        raise NotImplementedError()
      # invert the min-max normalization for continuous data
      if self.num_cat_index  < len(self.mask_cat_bf):
          solution[:,~np.array(self.mask_cat_bf)] = self.scaler.inverse_transform(solution[:,~np.array(self.mask_cat_bf)])  
      #Remove dummy encoding if there was
      if len(self.mask_cat_bf)!=len(self.mask_cat) and self.one_hot_encoding:
        solution =self.clean_onehot_data(solution)           
      # clip to min/max values
      solution= self.clip_extremes(solution) # this can be the cause of ValueError: XA and XB must have the same number of columns (i.e. feature dimension.)
      # Concatenate generated data with random label if needed if label conditioning was used 
      if self.label_y is not None:
        solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1) 
      return solution
   
# Check the category type 
def check_categorical_columns(X, categorical_columns):
      for col in categorical_columns:
          unique_values = np.unique(X[:, col])  # Find unique values in the column
          N = unique_values.max()  # Maximum category value
          expected_values = np.arange(0, N + 1)  # Expected range [0, ..., N]
          assert np.array_equal(unique_values, expected_values), \
              f"Column {col} has missing categories or values outside the range [0, ..., {N}]. Use label encoding if necessary."

# print(f"Column {col}: {len(unique_values)} unique categories, correctly in the range [0, ..., {N}]")  
# Seperate dataset into multiple minibatches for memory-efficient training 
class IterForDMatrix(xgb.core.DataIter):
    """A data iterator for XGBoost DMatrix.
    `reset` and `next` are required for any data iterator, the other functions here
    are utilites for demonstration's purpose.
    mask_cat and and get_xt_y are respectively functions to concatenate the input of the velocity vector and get a conditional flow at time t
    """
    def __init__(self, data,mask_cat,n_t,get_xt_y,make_mat,t,i, dim,model_type="HS3F",   n_batch=1000, n_epochs=10, eps=1e-3):
        self._data = data
        self.n_batch = n_batch
        self.n_epochs = n_epochs
        self.t = t
        self.make_mat=make_mat
        self.eps = eps
        self.dim = dim
        self.it = 0  # set iterator to 0
        self.n_t=n_t
        self.i=i
        self.model_type=model_type
        self.mask_cat=mask_cat
        self.get_xt_y=get_xt_y
        super().__init__()

    def reset(self):
      """Reset the iterator"""
      self.it = 0

    def next(self, input_data):
      """Yield next batch of data."""
      if self.it == self.n_batch*self.n_epochs: # stops after k epochs
        return 0
      x1=self._data[self.it % self.n_batch]
      if self.dim==0:
        if not self.mask_cat[self.dim]: 
          X_t,y= self.get_xt_y(x1,self.dim,self.t,self.i) 
          y_no_miss = ~np.isnan(y.ravel())
          input_data(data=X_t[y_no_miss, :], label=y[y_no_miss])
          self.it += 1  
        #If this is not respected then we move because the first categorical variable is generated using a multinomial sampling based on the class frequencies of this variable 
      else:
        if self.mask_cat[self.dim] and self.model_type== "HS3F": 
          x_t,y= self.make_mat(x1,self.dim),x1[:,self.dim]   
        else:
          X_t,y= self.get_xt_y(x1,self.dim,self.t,self.i) 
          x_t= self.make_mat(x1,self.dim,X_t) 
        y_no_miss = ~np.isnan(y.ravel())
        input_data(data=x_t[y_no_miss, :], label=y[y_no_miss])
        self.it += 1
      return 1
     