import unittest
import sys
import pandas as pd
from My_package.Training_Functions import Training
from My_package.Data_loading import data_loading
from My_package.Flow_matching_class import CFM
from My_package.Solver_Functions import solvers
from My_package.Scaling_and_Clipping import Data_processing_functions
from My_package.Sampling_Functions import sampling
from My_package.Metrics import test_on_multiple_models, Metrics,compute_coverage

# def create_csv(data_set_name,dic):
#     dta=dic[list(dic.keys())[0]][0]
#     Metric_dtfr=pd.DataFrame(columns=dta.columns)
#     cpt=0
#     for i in data_set_name:
#         dt=dic[i][0]
#         Metric_dtfr.loc[cpt]=dt.iloc[0]
#         cpt+=1
#     return Metric_dtfr

# class TestTrainingClass(unittest.TestCase):
#     def test_training_method(self):
#         data_set_name=["iris","wine","congress","tic-tac-toe","heart_disease", "ionosphere", "breast_cancer"]
#         FM_instance = CFM(sigma=0.0) 
#         Metrics4_data={}
#         ngen,nexp,cat_sampler_type,model_type=1,1,"proba_based","cont&cat"

#         N,K_dpl,Which_solver,problem_type,Use_OneHotEnc=20,30,"Euler", ["Class","Class","Class","Class","Class","Class","Class"],True
#         # data_set=sys.argv[2]
#         # if data_set=="Website_data":
#         for i in range(len(data_set_name)):
#                 if data_set_name[i] in ["heart_disease"]:
#                     dt=data_loading(data_set_name[i]).data_loader()
#                     Metrics4_data[data_set_name[i]]=Metrics(ngen,nexp,sampling,dt,data_set_name[i],
#                                         N,K_dpl,Which_solver,model_type,Use_OneHotEnc,cat_sampler_type,problem_type[i],forest_flow=None,mask_cat=None)
#                     Metrics4_data[data_set_name[i]][0].to_csv(f'/Users/ange-clementakazan/Documents/My_metrics/Metric_{model_type}_Euler_{data_set_name[i]}Use_OneHotEnc={Use_OneHotEnc}.csv',mode='w')
        # else:        #      for j in
def create_csv(j,perc,dic):
    dta=dic[j][list(dic[j].keys())[0]][0]
    Metric_dtfr=pd.DataFrame(columns=dta.columns)
    cpt=0
    for p in perc:
        dt=dic[j][f"{p*100}% of cat"][0]
        Metric_dtfr.loc[cpt]=dt.iloc[0]
        cpt+=1
    return Metric_dtfr
class TestTrainingClass(unittest.TestCase):
    def test_training_method(self):
        # data_set_name=["Maj_sum>0","Maj_vot","XORofAllSign"]
        FM_instance = CFM(sigma=0.0) 
        Metrics4_data={}
        ngen,nexp,cat_sampler_type,model_type,data_set_name=1,1,"model_prediction_based","cont_only", ["Maj_sum>0","Maj_vote>0","XORofAllSign"]
        perct_cat,num_samples,num_features=[i/10 for i in range(11)],2000,10
        N,K_dpl,Which_solver,Use_OneHotEnc=50,1,"Rg4",False
        # data_set=sys.argv[2]
        # if data_set=="Website_data":
        for j in data_set_name: 
                Metrics4_data[j]={}
                for p in perct_cat:
                            problem_type="Class"
                            dt=data_loading(j,p,num_samples,num_features)
                            rd_dt=dt.random_data_loader()
                            # print(rd_dt[0][:10][:,-1], rd_dt[-1])
                            # print(rd_dt[0])
                            Metrics4_data[j][f"{p*100}% of cat"]=Metrics(ngen,nexp,sampling,rd_dt,f"{j}__{p*100}% Dis",
                                                N,K_dpl,Which_solver,model_type,Use_OneHotEnc,cat_sampler_type,problem_type,forest_flow=None,mask_cat=None)
                            Metrics4_data[j][f"{p*100}% of cat"][0].to_csv(f'/Users/ange-clementakazan/Documents/My_metrics/Special_Metrics/Cat_pert:{p}_Metric_{model_type}_{Which_solver}_{data_set_name}Use_OneHotEnc={Use_OneHotEnc}.csv',mode='w')
                    
                Metric_dt=create_csv(j,perct_cat,Metrics4_data)
        # #Save it as csv
                Metric_dt.to_csv(f'/Users/ange-clementakazan/Documents/My_metrics/General_Metrics/Data:{j}{model_type}_{cat_sampler_type}_1HE:{Use_OneHotEnc}cont_cat_{Which_solver}.csv',mode='w')
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
