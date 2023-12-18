import pandas as pd
from My_package.Training_Functions import Training
from My_package.Data_loading import data_loader
from My_package.Flow_matching_class import CFM
from My_package.Solver_Functions import solvers
from My_package.Scaling_and_Clipping import Data_processing_functions
from My_package.Sampling_Functions import sampling
from My_package.Metrics_Functions import test_on_multiple_models, Metrics,compute_coverage

# def create_csv(data_set_name,dic):
#     dta=dic[list(dic.keys())[0]][0]
#     Metric_dtfr=pd.DataFrame(columns=dta.columns)
#     cpt=0
# #     for i in data_set_name:
#     dt=dic["iris"][0]
#     Metric_dtfr.loc[cpt]=dt.iloc[0]
#     cpt+=1
#     return Metric_dtfr
class Categorical_surgeon:
    def __init__(self, data, data_name)
    self.data=data
    self.data_name=data_name
    
    def Run_2F2S_Incrementally(self):
        
#         dic_4_newdata_name= { for i in range(self.data.shape[1])}
        Incremental_data=None
        Metrics4_data={}
        FM_instance = CFM(sigma=0.0) 
        ngen,nexp=1,1
        N,K_dpl,Which_solver=2,1,"Euler"  
        for k in range(1,self.data.shape[1]+1):
            Incremental_data=self.data[:,:k]
            New_data_name=f"{self.data_name}_{k}"
            
            Metrics4_data[k]=Metrics(ngen,nexp,sampling,Incremental_data,New_data_name,
                                     N,K_dpl,Which_solver,forest_flow=False,mask_cat=None)
        return Metrics4_data
    
data_set_name=["congress","tic-tac-toe"]            
 
Dict_4_Metrics={}
for i in data_set_name:
    Dict_4_Metrics[i]=Categorical_surgeon(data_loader(i),i).Run_2F2S_Incrementally()



Dict_4_Metric=create_csv(data_set_name,Dict_4_Metrics)
        #Save it as csv
Dict_4_Metric.to_csv('/Users/ange-clementakazan/Documents/Metrics_4_Forest_Flow_Based_Variable_Sampling_for incremental_data.csv')
        
        
if __name__ == '__main__':
    unittest.main()
            
            