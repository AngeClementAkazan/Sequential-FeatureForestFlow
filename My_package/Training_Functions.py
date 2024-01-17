" This script compute the forward process for the flow matching process, the training model for continuous/categorical variable  and the training process"

import numpy as np
import os
import copy
import xgboost as xgb
from numpy import loadtxt
from xgboost import XGBClassifier
from functools import partial
from sklearn.preprocessing import MinMaxScaler


class Training():
    def __init__(self,model,dt_loader,mask_cat,model_type,K_dpl,n_t,
n_estimators=100,
learning_rate=0.3,
tree_method='hist',
reg_lambda=1.0,
reg_alpha=0.0,
subsample=1.0, 
eta=0.3,  
max_depth=7,  
colsample_bytree=0.8, 
gamma=0,  
min_child_weight=1,  
scale_pos_weight=1,  
random_state=42):

        self.model=model
        self.dt_loader= dt_loader
        self.model_type=model_type
        self.mask_cat=mask_cat
        self.n_t=n_t
        self.K_dpl=K_dpl
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.eta=eta
        self.learning_rate=learning_rate
        self.tree_method=tree_method
        self.reg_lambda=reg_lambda
        self.reg_alpha=reg_alpha
        self.subsample=subsample 
        self.random_state=random_state
    
    def forward_process(self,dt_loader,model,n_t):  
        self.model=model
        self.dt_loader= dt_loader
        self.n_t=n_t
        b, c = self.dt_loader.shape
        X0 = np.random.normal(size=(b,c))        
        X_train = np.zeros((c,self.n_t, b,1))              # [c,n_t, b*100, 1]  # Will contain the interpolation between x0 and x1 (xt)   
        
        y_train = np.zeros((c,self.n_t,b, 1))               # [c,n_t, b*100, 1]  # Will contain the output to predict (ut).reshape(-1,1
        t_train = np.linspace(1e-9, 1, num=self.n_t)       
        for j in range(c):                                   # Fill the containers previously initialized with xt and ut
            for i in range(self.n_t):
                t = np.ones(self.dt_loader.shape[0])*t_train[i] # current t
                _, xt, ut =  self.model.Extr_CFM(X0[:,j].reshape(-1,1), self.dt_loader[:,j].reshape(-1,1), t=t)
                X_train[j][i][:], y_train[j][i][:] = xt, ut
        return X_train, y_train,self.dt_loader


    def train_cont(self,X_train, y_train):
        model = xgb.XGBRegressor(n_estimators=self.n_estimators, objective='reg:squarederror', eta=0.107, max_depth=self.max_depth,reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, subsample=self.subsample, seed=666, tree_method=self.tree_method, device='cpu')
        y_no_miss = ~np.isnan(y_train.ravel())
        model.fit(X_train[y_no_miss,:], y_train[y_no_miss])
        return model

    def train_cat(self,X_train, y_train ):
        model = XGBClassifier(n_estimators=self.n_estimators,
            objective='multi:softmax' if len(np.unique(y_train)) > 2 else 'binary:logistic',
            eta=0.307,
            learning_rate=0.202,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            subsample=self.subsample,
            tree_method=self.tree_method,
            use_label_encoder=False,  # For XGBoost 1.3.0 and above
            eval_metric='mlogloss' if len(np.unique(y_train)) > 2 else 'logloss',
            seed=666,
            device='cpu')
        y_no_miss = ~np.isnan(y_train.ravel())
        model.fit(X_train[y_no_miss,:], y_train[y_no_miss])
        return model


    def training(self):
        x0_uniques = x0_probs = None
        results_cont = []           #Is the list that will contain the models for continuous variables
        results_cat = []            #Is the list that will contain the models for categorical variables
        f, g,X = self.forward_process(self.dt_loader,self.model,self.n_t)
        b, c = X.shape
        if self.model_type=="cont&cat":  #To choose mixture of model
            for k in range(c):
                if self.mask_cat[k]:  #if the categorical mask is true then we do the following operations
                        if k==0:
                            x0_uniques, x0_probs = np.unique(X[:,k].reshape(-1,1),return_counts=True)
                            x0_probs = x0_probs/np.sum(x0_probs)
                            x0_2get=x0_uniques[np.argmax(np.random.multinomial(1, x0_probs, size=(b,)), axis=1)]
                            results_cat.append(x0_2get)
                        else:
                            A=()
                            for h in range(k):
                                A+=(X[:,h].reshape(-1,1),)  # A is initialized with the first variable  and will contain all the variable precedenting the one we want to predict using our classifier
                            Yy=X[:,k]
                            X_train_chunks=np.concatenate(A,axis=1)
                            result = self.train_cat(X_train_chunks,Yy)
                            results_cat.append(result)
                else:                 #if the categorical mask is False then we do the following operations
                    for i in range(self.n_t):             
                        if k==0:
                            X_train_chunk,y_train_chunk= f[k][i], g[k][i]
                            result = self.train_cont(X_train_chunk, y_train_chunk)
                        else:
                            X_train_chunk,y_train_chunk= f[k][i], g[k][i]
                            A=(X_train_chunk,)  # A is initialized with a noise and will contain all the variable precedenting the one we want to predict using our classifier
                            for h in range(k):
                                A+=(X[:,h].reshape(-1,1),)
                            X_train_chunks=np.concatenate(A,axis=1)
                            result = self.train_cont(X_train_chunks,y_train_chunk)
                        results_cont.append(result)

            cont_list=[i for i in range(len(self.mask_cat)) if not self.mask_cat[i]]    #Get all the indices for continuous variable
            regr_ = [[None  for i in range(self.n_t)]  for p in range(len(cont_list))]   #Initialize a container that will receive the Xgboost Regressors build to predict noisy outputs

            current_i_cont = 0
            for kk in range(len(cont_list)):
                    for i in range(self.n_t):
                            regr_[kk][i] = results_cont[current_i_cont]
                            current_i_cont += 1
                            
        elif self.model_type== "cont_only":          #To choose only model for continuous settings
            for k in range(c):  
                for i in range(self.n_t):             
                        if k==0:
                            X_train_chunk,y_train_chunk= f[k][i], g[k][i]
                            result = self.train_cont(X_train_chunk, y_train_chunk)
                        else:
                            X_train_chunk,y_train_chunk= f[k][i], g[k][i]
                            A=(X_train_chunk,)  # A is initialized with a noise and will contain all the variable precedenting the one we want to predict using our classifier
                            for h in range(k):
                                A+=(X[:,h].reshape(-1,1),)
                            X_train_chunks=np.concatenate(A,axis=1)
                            result = self.train_cont(X_train_chunks,y_train_chunk)
                        results_cont.append(result)

             


            regr_ = [[None  for i in range(self.n_t)]  for k in range(c)  ]
            current_i_cont = 0
            for kk in range(c):
                for i in range(self.n_t):
                        regr_[kk][i] = results_cont[current_i_cont]
                        current_i_cont += 1
        else:
            print( " Choose the right value for the  model_type argument")
        
        return regr_,results_cat
       

        