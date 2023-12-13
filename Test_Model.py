import unittest
import pandas as pd
from My_package.Training_Functions import Training
from My_package.Data_loading import data_loader
from My_package.Flow_matching_class import CFM
from My_package.Solver_Functions import solvers
from My_package.Scaling_and_Clipping import Data_processing_functions
from My_package.Sampling_Functions import sampling
from My_package.Metrics_Functions import test_on_multiple_models, Metrics,compute_coverage
# import numpy as np
# from sklearn.datasets import load_iris,load_wine
def create_csv(data_set_name,ls):
    dta=ls["iris"][0]
    Metric_dtfr=pd.DataFrame(columns=dta.columns)
    cpt=0
    for i in data_set_name:
        dt=ls[i][0]
        Metric_dtfr.loc[cpt]=dt.iloc[0]
        cpt+=1
    return Metric_dtfr
class TestTrainingClass(unittest.TestCase):
    def test_training_method(self):
        data_set_name=["iris","wine","congress","tic-tac-toe","heart_disease"]
#         dt_loader, mask_cat, n_t,K_dpl,Which_solver = data_loader(data_set_name[0])[0], data_loader(data_set_name[0])[-1], 5,None,"Euler"        
        FM_instance = CFM(sigma=0.0)  # Assuming CFM is a class in Flow_matching_class.py
        Metrics4_data={}
        ngen,nexp=5,3
        N,K_dpl,Which_solver=50,100,"Euler" #Noise level and K_duplicate
        for i in data_set_name:
                Metrics4_data[i]=Metrics(ngen,nexp,sampling,data_loader,data_set_name,i,
                                         N,K_dpl,Which_solver,forest_flow=False,mask_cat=None)
        Metric_dt=create_csv(data_set_name,Metrics4_data)
        Metric_dt.to_csv('/Users/ange-clementakazan/Documents/Variable_Sampling_Forest_Flow_Metrics.csv')
if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
# My_Package_FFVS/
# ├── My_package/
# │   ├── __init__.py
# │   ├── Training_Functions.py
# │   ├── Data_loading.py
#     ├── Flow_matching_class
# │   └── ...
# ├── Test_Training_function.py
#         print(dt_loader.shape, FM_instance)
#         Train_c = Training(FM_instance, dt_loader,mask_cat, n_t)
#         train_c= Train_c.training()  # Adjust the method name based on your actual implementation
# #         # Add your assertions based on the expected result
# #         print(np.array(result[0]).shape)
#         sol=solvers(dt_loader,train_c,mask_cat, n_t)
#         solution=sol.euler_solve(x_k=None)
#         print("solution.shape:",solution.shape)