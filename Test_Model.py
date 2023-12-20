import unittest
import pandas as pd
from My_package import *

def create_csv(data_set_name,dic):
    dta=dic[list(dic.keys())[0]][0]
    Metric_dtfr=pd.DataFrame(columns=dta.columns)
    cpt=0
    for i in data_set_name:
        if i=="congress" or i=="tic-tac-toe":
            dt=dic[i][0]
            Metric_dtfr.loc[cpt]=dt.iloc[0]
            cpt+=1
    return Metric_dtfr

class TestTrainingClass(unittest.TestCase):
    def test_training_method(self):
        data_set_name=["iris","wine","congress","tic-tac-toe","heart_disease"]
        FM_instance = CFM(sigma=0.0) 
        Metrics4_data={}
        ngen,nexp,model_type=3,5,"cont&cat"
        N,K_dpl,Which_solver,problem_type=30,10,"Euler", ["Class","Class","Class","Class","Reg"]
        for i in range(len(data_set_name)):
            if data_set_name[i]=="congress" or data_set_name[i]=="tic-tac-toe":
                Metrics4_data[data_set_name[i]]=Metrics(ngen,nexp,sampling,data_loader(data_set_name[i]),data_set_name[i],
                                         N,K_dpl,Which_solver,model_type,problem_type[i],forest_flow=False,mask_cat=None)
        Metric_dt=create_csv(data_set_name,Metrics4_data)
#         print(Metric_dt)
        #Save it as csv
        Metric_dt.to_csv('/Users/ange-clementakazan/Documents/Metrics_4_Forest_Flow_Based_Variable_Sampling_with__cont_models.csv',mode='w')
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