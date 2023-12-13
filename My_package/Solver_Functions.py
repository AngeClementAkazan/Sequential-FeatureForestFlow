" These functions are the solvers used for the sampling of our flow matching models"
import numpy as np

class solvers:

    def __init__(self,dt_loader,tr_container,mask_cat,N):
    
        self.dt_loader=dt_loader 
        self.tr_container=tr_container
        self.mask_cat=mask_cat 
        self.N=N
        
        """ dt_loader: is the dt_loader to be inputted after dummy encoding
            tr_container: is the tuple that contains the lists for both categorical and continuous XGBoost models
            mask_cat: is the mask for categorical data (list containing True for categorical and False for Continuous) after dummy encoding
            N: is the number of noise level we are dealing with 
        """
        
    def my_model_cont(self,tr_container,dt_loader,t,k,count,N,x_k, x_prev):
        if x_prev is not None:
            x = np.concatenate((x_k, x_prev), axis=1)
        else:
            x = x_k
        b,c=self.dt_loader.shape
        out = np.zeros((b,c)) # [b, c]
        i = int(round(t*(self.N-1)))
        out[:, k] = self.tr_container[0][count][i].predict(x)
        return out
    def my_model_cat(self,tr_container,dt_loader,k,cat_cont, x_prev):
        b,c=self.dt_loader.shape
        out = np.zeros((b,c))
        if x_prev is None and k==0:
            out[:, k] = self.tr_container[1][0][:b]# random sample
        else:
            out[:, k]=self.tr_container[1][cat_cont].predict(x_prev)
        return out

  # Simple Euler ODE solver 
    def euler_solve(self,x_k):
        b,c=self.dt_loader.shape
        h = 1 / (self.N-1)
        x_prev = None
        A=()
        cat_count=0        #cat_count  serves to pick the right model from the list for categorical variable during training
        cont_count=0       #cont_count argument serves to pick the right model from the array for continous data
        for k in range(c):
            if self.mask_cat[k]:
                x_k = self.my_model_cat(self.tr_container,self.dt_loader,k,cat_count, x_prev)[:,k].reshape(-1,1)
                cat_count+=1
            else:
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
        A=np.concatenate(A,axis=1) 
        return A

    def runge_kutta_solve(self,x_fake):
        b,c=self.dt_loader.shape
        h = 1 / (self.N-1)
        A=()
        x_prev = None
        cat_count=0
        cont_count=0
        for k in range(c):
            if self.mask_cat[k]:
                x_fake= self.my_model_cat(self.tr_container,self.dt_loader,k,cat_count, x_prev=x_prev)[:,k].reshape(-1,1)
                cat_count+=1
            else:
                t=0
                x_fake=np.random.normal(size=(b,1))
                for i in range(self.N-1):
                    k1 = h * self.my_model_cont(self.tr_container,self.dt_loader,t,k,cont_count,self.N,x_fake, x_prev=x_prev)[:,k].reshape(-1,1)
                    k2 = h * self.my_model_cont(self.tr_container,self.dt_loader,t + h / 2,k,cont_count,self.N, x_fake + k1 / 2, x_prev=x_prev)[:,k].reshape(-1,1) 
                    k3 = h * self.my_model_cont(self.tr_container,self.dt_loader,t + h / 2,k,cont_count,self.N, x_fake + k2 / 2, x_prev=x_prev)[:,k].reshape(-1,1)
                    k4 = h * self.my_model_cont(self.tr_container,self.dt_loader,t + h,k,cont_count,self.N, x_fake + k3, x_prev=x_prev)[:,k].reshape(-1,1)
                    x_fake = x_fake + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    t = t + h
                cont_count+=1
            if x_prev is None:
                x_prev = x_fake
            else:
                x_prev = np.concatenate((x_prev,x_fake), axis=1)
            A+=(x_fake ,)
    #         print(A.shape)
        A=np.concatenate(A,axis=1)
        return A             
                