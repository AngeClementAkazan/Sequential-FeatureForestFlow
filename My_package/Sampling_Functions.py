import numpy as np
from sklearn.preprocessing import MinMaxScaler
from My_package.utils.Training_Functions import Training
# from utils import *
from My_package.utils.Flow_matching_class import CFM
from My_package.utils.Solver_Functions import solvers
from My_package.utils.Scaling_and_Clipping import Data_processing_functions

FM_instance = CFM(sigma=0.0)                   
class sampling:
                 
    def __init__(self,dt_loader,mask_cat,N,K_dpl,model_type,Use_OneHotEnc,cat_sampler_type,which_solver=None,arg1={},arg2={}):
        self.dt_loader=dt_loader 
        self.mask_cat=mask_cat
        self.N=N
        self.K_dpl=K_dpl
        self.which_solver=which_solver 
        self.model_type=model_type 
        self.cat_sampler_type=cat_sampler_type
        self.Use_OneHotEnc= Use_OneHotEnc
        self.arg1=arg1
        self.arg2=arg2

        """ dt_loader: is the data to be inputted
            K_dpl: is the number of time we duplicate our data
            mask_cat: is the mask for categorical data (list containing True for categorical and False for Continuous
            N: is the number of noise level we are dealing with 
            which_solver: takes two values: {Euler: for Euler solver or RG4: for Runge Kutta solver}
            model_type: specifies whether we have a mixed model (regressor and classification) or regressor only 
            cat_sampler_type: determine whether we use the Xgboost model prediction directly for sampling(in that case the argument take the value "model_prediction-based") or we use the output probability of our Xgboost and then use a multinoimial sampler(and the argument take "proba-based")
            Use_OneHotEnc: Determine whether or not we will use one hot encoding (takes argument True or False)
            arg1 and arg2 are respectively, the remaining hyperparameter for tunning the regressor and the classifier ( We did not consider all the argument for our Xgboost regressor and classifier, ythe user will define them personnally if needed)
       """
    # kwarg1=list(kwargs.items)
   
    def Final_training(self,data, N, K_dpl):
            cat_ind_b4_1hotEnc=[id for id in range(len(self.mask_cat)) if self.mask_cat[id]] 
            X_transformed=data
            X_names_before, X_names_after,scaler, mask_cat_4_1hotEnc= None,None,None,self.mask_cat
            if len(cat_ind_b4_1hotEnc) > 0 and self.Use_OneHotEnc== True:
                X_transformed, X_names_before, X_names_after,mask_cat_4_1hotEnc = Data_processing_functions.dummify(data,self.mask_cat)
                
            # print("Xtransformed.shape:",X_transformed.shape, self.K_dpl)
            if len(cat_ind_b4_1hotEnc)<data.shape[1]:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                X_transformed[:,~np.array(mask_cat_4_1hotEnc)]=scaler.fit_transform(X_transformed[:,~np.array(mask_cat_4_1hotEnc)])

            data_4_training=np.tile(X_transformed, (self.K_dpl, 1))
            Train_c=Training(FM_instance,data_4_training,mask_cat_4_1hotEnc,self.model_type,self.K_dpl,self.N,self.arg1,self.arg2)
            train_c=Train_c.training()     #Container that hold the training Xgboost model
            return train_c,X_transformed,data_4_training,scaler, cat_ind_b4_1hotEnc,mask_cat_4_1hotEnc,X_names_before, X_names_after

    def sample(self):    
            dta,msk_ct=self.dt_loader[0],self.dt_loader[-1]
             # Sanity check, must remove observations with only missing data
            obs_to_remove = np.isnan(dta).all(axis=1)
            dta = dta[~obs_to_remove]
            train_c,dt_Aftr_or_Not_1hE,data_4_training,scaler, cat_ind_b4_1hotEnc,mask_cat_4_1hotEnc,X_names_before, X_names_after=self.Final_training(dta,self.N,self.K_dpl)  #Training container, dummified and transformed data plus the new mask 
            x_k=x_fake= None
            ### ODE solvers ###
            Solver=solvers(dt_Aftr_or_Not_1hE,train_c,self.cat_sampler_type,mask_cat_4_1hotEnc,self.model_type,self.N)
            if self.which_solver == "Euler":
                solution = Solver.euler_solve(x_k) # Euler solver               
            elif self.which_solver == "Rg4":
                solution= Solver.runge_kutta_solve(x_fake) #Runge Kutta solver                 
            else:
                raise Exception("Kindly choose a solver between Euler or Runge Kutta")   
                     
            if len(cat_ind_b4_1hotEnc)<dta.shape[1]:  # invert the min-max normalization for continuous variable only if there is
                solution[:,~np.array(mask_cat_4_1hotEnc)] = scaler.inverse_transform(solution[:,~np.array(mask_cat_4_1hotEnc)])                   
            #Remove dummy encoding if there was
            if len(msk_ct)!=len(mask_cat_4_1hotEnc):
                 solution = Data_processing_functions.clean_onehot_data(solution, X_names_before, X_names_after,mask_cat_4_1hotEnc)
            #Save min/max of the values of the real data
            dt=self.dt_loader[0]
            X_min = np.nanmin(dt, axis=0, keepdims=1)
            X_max = np.nanmax(dt, axis=0, keepdims=1)
            # clip to min/max values
            solution=Data_processing_functions.clipping(X_min,X_max,solution,dta,msk_ct)
            return solution