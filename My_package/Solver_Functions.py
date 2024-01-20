" These functions are the solvers used for the sampling of our flow matching models"
" This script describe the sampling functions we have used for our model implementation"
import numpy as np

class solvers: 
    def __init__(self,dt_loader,tr_container,cat_sampler_type,mask_cat,model_type,N):
        self.dt_loader=dt_loader 
        self.tr_container=tr_container
        self.mask_cat=mask_cat 
        self.N=N 
        self.model_type=model_type
        self.cat_sampler_type =cat_sampler_type
        """ dt_loader: is the data to be inputted after dummy encoding
            tr_container: is the tuple that contains the lists for both categorical and continuous XGBoost models
            mask_cat: is the mask for categorical data (list containing True for categorical and False for Continuous) after dummy encoding (depending on whether cat_sampler_type is model-predict-based)
            N: is the number of noise level we are dealing with
            cat_sampler_type: determine whether we use the Xgboost model prediction directly for sampling(in that case the argument take the value "model_prediction-based") or we use the output probability of our Xgboost and then use a multinoimial sampler(and the argument take "proba-based")
       """
       
    def my_model_cont(self,tr_container,dt_loader,t,k,count,N,x_k, x_prev):
        if x_prev is None:                
            x = x_k    #If no previous variable (x_prev==None), x is the noisy input data used to generate the first variable of the data
        else:         # x receives the previous variable having been 
            x = np.concatenate((x_k, x_prev), axis=1) # We respect the training structure for continuous variable that is: the model reveives (X_noise, Variable1,...,Variable k-1) to predict Variable k
        b,c=self.dt_loader.shape
        out = np.zeros((b,c)) # [b, c]
        i = int(round(t*(self.N-1)))
        out[:, k] = self.tr_container[0][count][i].predict(x)
        return out
 
    def my_model_cat(self,tr_container,dt_loader,k,cat_count, x_prev):
        b,c=self.dt_loader.shape
        out = np.zeros((b,c))
        if x_prev is None and k==0:
            out[:, k] = self.tr_container[1][0]# random sample
        else:
            if self.cat_sampler_type=="model_prediction_based":
                out[:, k]=tr_container[1][cat_count].predict(x_prev)

            elif self.cat_sampler_type=="proba_based":
                x_pred=tr_container[1][cat_count].predict_proba(x_prev)
                x_fake = np.zeros(b)
                y_uniques,y_count=np.unique(self.dt_loader[:,k],return_counts=True)
                for j in range(b):
                    x_fake[j] = y_uniques[np.argmax(np.random.multinomial(1, x_pred[j], size=1), axis=1)] # sample according to probability
                out[:, k] =x_fake     
            else:
                raise Exception("Choose the right sampling mode")                
        return out

  # Simple Euler ODE solver 
    def euler_solve(self,x_k):
        b,c=self.dt_loader.shape
        h = 1 / (self.N-1)
        x_prev = None      #x_prev is the container that will receive all the predicted variables
        A=()
        cat_count=0        #cat_count  is used to pick the right model from the list for categorical variable during training
        cont_count=0       #cont_count argument is used to pick the right model from the array for continous data   
        if self.model_type== "cont&cat":
            for k in range(c):
                if self.mask_cat[k]:
                    x_k = self.my_model_cat(self.tr_container,self.dt_loader,k,cat_count, x_prev)[:,k].reshape(-1,1)
                    cat_count+=1
                else:
                    t=0
                    x_k=np.random.normal(size=(b,1))
                    for i in range(self.N-1):
                        x_k = x_k + h*self.my_model_cont(self.tr_container,self.dt_loader,t,k,cont_count,self.N,x_k, x_prev)[:,k].reshape(-1,1)                     #[:,k] because we want to return the k th column predicted by the model
                        t = t + h
                    cont_count+=1
                if x_prev is None:
                    x_prev = x_k     
                else:
                    x_prev = np.concatenate((x_prev,x_k), axis=1)
                A+=(x_k,)
        elif self.model_type== "cont_only":
            for k in range(c):
                t=0
                x_k=np.random.normal(size=(b,1))
                for i in range(self.N-1):
                    x_k = x_k + h*self.my_model_cont(self.tr_container,self.dt_loader,t,k,cont_count,self.N,x_k, x_prev)[:,k].reshape(-1,1)                     # k because we want to return the k th column preddicted by the model
                    t = t + h
                cont_count+=1
                if x_prev is None:
                    x_prev = x_k
                else:
                    x_prev = np.concatenate((x_prev,x_k), axis=1)
                A+=(x_k,)
        else:
            raise Exception("Choose the model type as: {cont&cat: for mixed regressor and classifier model}, cont_only: for regressor only}, This has an impact on the Solver_Functions module")
        A=np.concatenate(A,axis=1) 
        return A

    def runge_kutta_solve(self,x_fake):
        b,c=self.dt_loader.shape
        h = 1 / (self.N-1)
        A=()
        x_prev = None
        cat_count=0
        cont_count=0
        if self.model_type== "cont&cat":
            for k in range(c):
                if self.mask_cat[k]:
                    x_fake = self.my_model_cat(self.tr_container,self.dt_loader,k,cat_count, x_prev)[:,k].reshape(-1,1)
                    cat_count+=1
                else:
                    t=0
                    x_fake=np.random.normal(size=(b,1))
                    for i in range(self.N-1):
                        x_fake = x_fake + h*self.my_model_cont(self.tr_container,self.dt_loader,t,k,cont_count,self.N,x_fake, x_prev)[:,k].reshape(-1,1)                     #[:,k] because we want to return the k th column predicted by the model
                        t = t + h
                    cont_count+=1
                if x_prev is None:
                    x_prev =x_fake     
                else:
                    x_prev = np.concatenate((x_prev,x_fake), axis=1)
                A+=(x_fake,)
        elif self.model_type== "cont_only":
            for k in range(c):
                t=0
                x_fake=np.random.normal(size=(b,1))
                for i in range(self.N-1):
                    x_fake =x_fake + h*self.my_model_cont(self.tr_container,self.dt_loader,t,k,cont_count,self.N,x_fake, x_prev)[:,k].reshape(-1,1)                     # k because we want to return the k th column preddicted by the model
                    t = t + h
                cont_count+=1
                if x_prev is None:
                    x_prev = x_fake
                else:
                    x_prev = np.concatenate((x_prev,x_fake), axis=1)
                A+=(x_fake,)
        else:
            raise Exception("Choose the model type as: {cont&cat: for mixed regressor and classifier model}, cont_only: for regressor only},This has an impact on the Solver_Functions module ")
        A=np.concatenate(A,axis=1)
        return A             
                