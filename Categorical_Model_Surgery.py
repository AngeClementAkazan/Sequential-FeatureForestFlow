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
    for i in range(1,len(data_set_name)):
        dt=dic[i][0]
        Metric_dtfr.loc[cpt]=dt.iloc[0]
        cpt+=1
    return Metric_dtfr

class Categorical_surgeon:
    def __init__(self, dt_loader, data_name):
        self.dt_loader=dt_loader
        self.data_name=data_name
    
    def Run_2F2S_Incrementally(self):
        Incremental_data=None
        Metrics4_data={}
        FM_instance = CFM(sigma=0.0) 
        ngen,nexp,model_type=3,5,"cont&cat"
        N,K_dpl,Which_solver,problem_type=50,100,"Rg4","Class"
        for k in range(1,self.dt_loader[0].shape[1]+1):
            Incremental_data=self.dt_loader[0][:,:k]
            mask_cat=self.dt_loader[-1][:k]
            dt_loader_Inc=(Incremental_data,mask_cat)
            
            New_data_name=f"{self.data_name}_{k}"
            
            Metrics4_data[k]=Metrics(ngen,nexp,sampling,dt_loader_Inc,New_data_name,
                                         N,K_dpl,Which_solver,model_type,problem_type,forest_flow=False,mask_cat=None)
        Metric_dt=create_csv(data_set_name,Metrics4_data)
        return Metrics4_data
    data_set=["congress","tic-tac-toe"]            
# data_set=["congress"]  
Dict_4_Metrics={}
for i in range(len(data_set)):
    data_set_name=[f"{data_set[i]}_{k}" for k in range(1,data_loader(data_set[i])[0].shape[1]+1) ]
    Dict_4_Metrics[i]=Categorical_surgeon(data_loader(data_set[i]),data_set[i]).Run_2F2S_Incrementally()
#     print(Dict_4_Metrics[i])
    Classsifier_surgery=create_csv(data_set_name,Dict_4_Metrics[i])
    Classsifier_surgery.to_csv(f"/Users/ange-clementakazan/Documents/Metrics_for_Forest_Flow_Based_Variable_Sampling_for_{data_set[i]}_incremental_data.csv",mode='w')
    print(f" The {data_set[i]} data set has been treated")


            
            