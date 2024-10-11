import unittest
import sys
import pandas as pd
# import numpy as np
from Seq3F.Metrics import Metrics
from Seq3F.S3F import feature_forest_flow
from ForestFlow import ForestDiffusionModel #Forest Flow

# np.random.seed(999)
# To run this script, in your terminal run this:  python -m unittest Test_Model_.Test_class.test

#Create a function that will create a csv table for  the error metrics for the data set of test_method_2
def create_csv(data_set_name,dic):
    dta=dic[list(dic.keys())[0]][0]
    Metric_dtfr=pd.DataFrame(columns=dta.columns)
    cpt=0
    for i in data_set_name:
        dt=dic[i][0]
        Metric_dtfr.loc[cpt]=dt.iloc[0]
        cpt+=1
    return Metric_dtfr

class Test_class(unittest.TestCase):
    """ K_dpl: is the number of time we duplicate our data
        mask_cat: is the mask for categorical data (list containing True for categorical and False for Continuous)
        N: is the number of noise level we are dealing with 
        which_solver: takes two values: {Euler: for Euler solver or RG4: for Runge Kutta solver}
        model_type: specifies whether we have a mixed model (regressor and classification) or regressor only 
        prediction_type: determine whether we use the Xgboost model prediction directly for sampling(in that case the argument take the value "model_prediction-based") or we use the output probability of our Xgboost and then use a multinoimial sampler(and the argument take "proba-based")
        one_hot_encoding: Determine whether or not we will use one hot encoding (takes argument True or False)
        arg1 and arg2 are respectively, the remaining hyperparameter for tunning the regressor and the classifier ( We did not consider all the argument for our Xgboost regressor and classifier, ythe user will define them personnally if needed)
        n_batch: is the number of mini batch 
        n_jobs: specifies the number jobs you wish to exucute with your computing cores (-1 uses everything possible)
        Sequential_Feature_forest_flow: Bolean value that specifies whether we use S3F when set to true and ForestFlow otherwise
    """
    #Test script for collecting the results from all the datasets used in this study
    def test(self):     
        data_set_name= ['iris', 'wine','parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast','libras',  'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel', 'concrete_slump', \
            'wine_quality_red', 'wine_quality_white', \
             'tictactoe','congress','car'] 
        model=feature_forest_flow 
        parent="/Users/ange-clementakazan/Documents/My_metrics/FFM" # Change this 
        # data_set_name= ['iris']
        Metrics4_data={}
        ngen,nexp,model_type,label_cond=1,1,"HS3F",True
        arg1,arg2,n_batch,n_jobs={},{},0,-1 
        N,K_dpl,which_solver=50,100,"Euler"
        solver=["Rg4","Euler"]
        model_type_=["HS3F","CS3F"]
        for model_type in model_type_:
            for which_solver in solver:
                for name in data_set_name:
                    Metrics4_data[name]=Metrics(ngen,nexp,model,name,
                                        N,K_dpl,which_solver,model_type,n_batch,n_jobs,label_cond,arg1,arg2,Sequential_Feature_forest_flow=True,mask_cat=None)
                    
                    Metrics4_data[name][0].to_csv(parent+f'/{model_type}_{which_solver}_{name}.csv',mode='w')
                Metric_dt=create_csv(data_set_name,Metrics4_data)
                #Save it as csv in Your folder #
                Metric_dt.to_csv(parent+f'/S3F_{model_type}_{which_solver}.csv',mode='w')
if __name__ == '__main__':
    unittest.main()
    


   
