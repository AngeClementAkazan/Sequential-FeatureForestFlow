import unittest
import pandas as pd
from My_package.Training_Functions import Training
from My_package.Data_loading import data_loader
from My_package.Flow_matching_class import CFM
from My_package.Solver_Functions import solvers
from My_package.Scaling_and_Clipping import Data_processing_functions
from My_package.Sampling_Functions import sampling
from My_package.Metrics import test_on_multiple_models, Metrics,compute_coverage

def create_csv(data_set_name,dic):
    dta=dic[list(dic.keys())[0]][0]
    Metric_dtfr=pd.DataFrame(columns=dta.columns)
    cpt=0
    for i in data_set_name:
        dt=dic[i][0]
        Metric_dtfr.loc[cpt]=dt.iloc[0]
        cpt+=1
    return Metric_dtfr

class TestTrainingClass(unittest.TestCase):
    def test_training_method(self):
        data_set_name=["iris","wine","congress","tic-tac-toe","heart_disease"]
        FM_instance = CFM(sigma=0.0) 
        Metrics4_data={}
        ngen,nexp,model_type=3,5,"cont_only"
        N,K_dpl,Which_solver,problem_type=50,30,"Rg4", ["Class","Class","Class","Class","Reg"]
        for i in range(len(data_set_name)):
            Metrics4_data[data_set_name[i]]=Metrics(ngen,nexp,sampling,data_loader(data_set_name[i]),data_set_name[i],
                                         N,K_dpl,Which_solver,model_type,problem_type[i],forest_flow=False,mask_cat=None)
        Metric_dt=create_csv(data_set_name,Metrics4_data)

        #Save it as csv
        Metric_dt.to_csv('/Users/ange-clementakazan/Documents/Metrics_4_Forest_Flow_Based_Variable_Sampling_with__RungeKutta_cont_models.csv',mode='w')
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
# __all__=["Data_loading", "Flow_matching_class","Metrics","Sampling_Functions","Scaling_and_Clipping","Solver_Functions","Training_Functions"]
