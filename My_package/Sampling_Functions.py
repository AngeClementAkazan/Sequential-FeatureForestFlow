import numpy as np
from sklearn.preprocessing import MinMaxScaler
from My_package.Training_Functions import Training
from My_package.Flow_matching_class import CFM
from My_package.Solver_Functions import solvers
from My_package.Scaling_and_Clipping import Data_processing_functions


"Flow matching class"
FM_instance = CFM(sigma=0.0)                   
class sampling:
                  
    def __init__(self,dt_loader,N,K_dpl,model_type,cat_sampler_type,which_solver=None):
        self.dt_loader=dt_loader 
        self.N=N
        self.K_dpl=K_dpl
        self.which_solver=which_solver 
        self.model_type=model_type 
        self.cat_sampler_type=cat_sampler_type
        """ dt_loader: is the data to be inputted
            K_dpl: is the number of time we duplicate our data
            mask_cat: is the mask for categorical data (list containing True for categorical and False for Continuous
            N: is the number of noise level we are dealing with 
            which_solver: takes two values: {Euler: for Euler solver or RG4: for Runge Kutta solver}
            model_type: specifies whether we have a mixed model (regressor and classification) or regressor only 
            cat_sampler_type: determine whether we use the Xgboost model prediction directly for sampling(in that case the argument take the value "model_prediction-based") or we use the output probability of our Xgboost and then use a multinoimial sampler(and the argument take "proba-based")
        """

    def Final_training(self,data,N,K_dpl,mask_cat):
            cat_ind_b4_1hotEnc=[ii for ii in range(len(mask_cat)) if mask_cat[ii]] 
            
            X_transformed=data
            if len(cat_ind_b4_1hotEnc) > 0 :
                X_transformed, X_names_before, X_names_after,mask_cat_4_1hotEnc = Data_processing_functions.dummify(data,mask_cat,drop_first=True)
            else:
                X_names_before, X_names_after,scaler, mask_cat_4_1hotEnc= None,None,None,mask_cat 
            if len(cat_ind_b4_1hotEnc)<data.shape[1]:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                X_transformed[:,~np.array(mask_cat_4_1hotEnc)]=scaler.fit_transform(X_transformed[:,~np.array(mask_cat_4_1hotEnc)])
                                   
            data_4_training=np.tile(X_transformed, (self.K_dpl, 1))
            Train_c=Training(FM_instance,data_4_training,mask_cat_4_1hotEnc,self.model_type,self.K_dpl,self.N)
            train_c=Train_c.training()     #Container that hold the training Xgboost model
            return train_c,X_transformed,data_4_training,scaler, cat_ind_b4_1hotEnc,mask_cat_4_1hotEnc,X_names_before, X_names_after

    def sample(self):    
            dta,msk_ct=self.dt_loader[0],self.dt_loader[-1]
             # Sanity check, must remove observations with only missing data
            obs_to_remove = np.isnan(dta).all(axis=1)
            dta = dta[~obs_to_remove]
            train_c,dt_Aftr_or_Not_1hE,data_4_training,scaler, cat_ind_b4_1hotEnc,mask_cat_4_1hotEnc,X_names_before, X_names_after=self.Final_training(dta,self.N,self.K_dpl,msk_ct)  #Training container, dummified and transformed data plus the new mask 
            x_k=x_fake= None
            # ODE solve
            Solver=solvers(data_4_training,train_c,self.cat_sampler_type,mask_cat_4_1hotEnc,self.model_type,self.N)
            if self.which_solver == "Euler":
                solution = Solver.euler_solve(x_k) # Euler solver               
            elif self.which_solver == "Rg4":
                solution= Solver.runge_kutta_solve(x_fake) #Runge Kutta solver                 
            else:
                raise Exception("Kindly choose a solver between Euler or Runge Kutta")            
            if len(cat_ind_b4_1hotEnc)<dta.shape[1]:  # invert the min-max normalization for continuous variable only if there is
                solution[:,~np.array(mask_cat_4_1hotEnc)] = scaler.inverse_transform(solution[:,~np.array(mask_cat_4_1hotEnc)])                   
            #Remove dummy encoding if there was
            if dt_Aftr_or_Not_1hE.shape[1]!=dta.shape[1]: 
                 solution = Data_processing_functions.clean_onehot_data(solution, X_names_before, X_names_after,mask_cat_4_1hotEnc)
             # Save min/max of the values
            X_min = np.nanmin(dta, axis=0, keepdims=1)
            X_max = np.nanmax(dta, axis=0, keepdims=1)
            # clip to min/max values
            solution=Data_processing_functions.clipping(X_min,X_max,solution,dta,msk_ct)
            return solution