
import time
import numpy as np
import pandas as pd
import ot as pot
import random
from Seq3F.data_loader_ import dataset_loader
from Seq3F.utils_ import minmax_scale_dummy,test_on_multiple_models,compute_coverage
from sklearn.model_selection import train_test_split



def Metrics(ngen,nexp,model,data_name,n_t,K_dpl,which_solver,model_type,
            n_batch,n_jobs,label_cond=False,arg1={},arg2={},perc_cat=None,
            num_samples=None,num_features=None,Sequential_Feature_forest_flow=True,
            mask_cat=None):
    """    
        K_dpl: is the number of time we duplicate our data
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
    OTLIM = 5000
    method=model_type 
    if not Sequential_Feature_forest_flow:
        method="FF"
    score_W1_train = {}
    score_W1_test = {}
    coverage_train = {}
    coverage_test = {}
    time_taken = {}
    # for method in methods:
    method_str =method
    score_W1_test[method] = 0.0
    score_W1_train[method] = 0.0
    coverage_train[method] = 0.0
    coverage_test[method] = 0.0
    time_taken[method] = 0.0
    R2 = {}
    f1 = {}
    # print(n_jobs)
    # Get data
    X_, cat_indexes,int_indexes, y, bin_y, cat_y, int_y= dataset_loader(data_name)
    X=np.concatenate((X_,y.reshape(-1,1)), axis=1)
    # Remove label conditioning for data sets car as 
    if data_name=='car':
        label_cond=False
    # print(X.shape)

    # try:
    #   int_indexes = int_indexes + bin_indexes # since we round those, we do not need to dummy-code the binary variables
    # except:
    #    int_indexes=[]

    # Define the problem type
    cat_y=cat_y or bin_y   
    if cat_y:
        problem_type="Class"      
    else:
        problem_type="Reg"
        if label_cond==True:
            label_cond=False
    # print(cat_y)
    # for method in methods:
    R2 = {'real': {}, 'fake': {}, 'both': {}}
    f1 = {'real': {}, 'fake': {}, 'both': {}}
    for test_type in ['real','fake','both']:
        for test_type2 in ['mean','lin','linboost', 'tree', 'treeboost']:
            R2[test_type][test_type2] = 0.0
            f1[test_type][test_type2] = 0.0
    if mask_cat== None:
        if cat_indexes is not None:
            if  cat_y or bin_y:        
                mask_cat= [i in cat_indexes  for i in range(X.shape[1]-1)]+[True]  #Correct
            else:
                mask_cat= [i in cat_indexes  for i in range(X.shape[1]-1)]+[False]  #Correct
        else:
            if  cat_y or bin_y:      
                mask_cat=[False]*(X.shape[1]-1) +[True]
            else:
                mask_cat=[False]*(X.shape[1])

    # Loop over the number of experiments you wish to run (nexp=1 by defaults)
    for n in range(nexp):
            # Split the data into train and test as we will use the train data for generation
            X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=n, stratify=y if cat_y else None)
            Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
            Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)
            start = time.time()
            if Sequential_Feature_forest_flow== True:
                if label_cond==True and cat_y:
                    forest_model = model(X=Xy_train[:,:-1], 
                            label_y=Xy_train[:,-1],
                            n_t=n_t, # number of noise level
                            model='xgboost', # xgboost, random_forest, lgbm, catboost,
                            solver_type=which_solver,
                            model_type=model_type, # HS3F for hybrid and  CS3F for regressor only
                            duplicate_K=K_dpl, # number of different noise sample per real data sample
                            # bin_indexes=bin_indexes, # vector which indicates which column is binary
                            cat_indexes=cat_indexes, #Vector indicating which column is categorical
                            int_indexes=int_indexes, # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
                            cat_y=cat_y, # Binary variable indicating whether or not the output is categorical
                            n_jobs=n_jobs, # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
                            n_batch=n_batch, # If >0 use the data iterator with the specified number of batches
                            ngen=ngen,
                            seed=n)
                else:                   
                    forest_model = model(X=Xy_train, 
                            n_t=n_t, # number of noise level
                            model='xgboost', # xgboost, random_forest, lgbm, catboost,
                            solver_type=which_solver,
                            model_type=model_type, # HS3F for hybrid and  CS3F for regressor only
                            duplicate_K=K_dpl, # number of different noise sample per real data sample
                            # bin_indexes=bin_indexes, # vector which indicates which column is binary
                            cat_indexes=cat_indexes, #Vector indicating which column is categorical
                            int_indexes=int_indexes, # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
                            cat_y=cat_y, # Binary variable indicating whether or not the output is categorical
                            n_jobs=n_jobs, # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
                            n_batch=n_batch, # If >0 use the data iterator with the specified number of batches
                            ngen=ngen,
                            seed=n)
                
                Xy_fake = forest_model.generate(batch_size=ngen*Xy_train.shape[0]) # Shuffle the observations!
                np.random.shuffle(Xy_fake)
                Xy_fake = Xy_fake.reshape(ngen, Xy_train.shape[0], Xy_train.shape[1]) #
            else:
                if label_cond and cat_y:
                        forest_model = model(X=Xy_train[:,:-1], 
                            label_y=Xy_train[:,-1],
                            n_t=n_t,
                            model='xgboost', # in random_forest, xgboost, lgbm
                            duplicate_K=K_dpl,
                            # bin_indexes=bin_indexes,
                            cat_indexes=cat_indexes,
                            int_indexes=int_indexes,
                            n_jobs=n_jobs,
                            n_batch=n_batch,
                            seed=n)
                else:
                    forest_model = model(X=Xy_train, 
                        n_t=n_t,
                        model='xgboost', # in random_forest, xgboost, lgbm
                        duplicate_K=K_dpl,
                        # bin_indexes=bin_indexes, # vector which indicates which column is binary
                        cat_indexes=cat_indexes, #Vector indicating which column is categorical
                        int_indexes=int_indexes, # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
                        n_jobs=n_jobs,
                        n_batch=n_batch,
                        seed=n)
                # if method== "model":
                Xy_fake = forest_model.generate(batch_size=ngen*Xy_train.shape[0],n_t=n_t) # Shuffle the observations!
                np.random.shuffle(Xy_fake)
                Xy_fake = Xy_fake.reshape(ngen, Xy_train.shape[0], Xy_train.shape[1]) #
            end = time.time()
            time_taken[method] += (end - start) / nexp 
            #Remove all  NAN values in the train data if there are 
            obs_to_remove_tr, obs_to_remove_te= np.isnan(Xy_train).any(axis=1), np.isnan(Xy_test).any(axis=1)
            Xy_train, Xy_test=Xy_train[~obs_to_remove_tr], Xy_test[~obs_to_remove_te]
            #Determining  a tensor of ngen generated samples based on the train X                   
            for gen_i in range(ngen):
                Xy_fake_i=Xy_fake[gen_i]
                Xy_train_scaled, Xy_fake_scaled,_,_,_,_= minmax_scale_dummy(Xy_train,Xy_fake_i,mask_cat,divide_by=2)
                _, Xy_test_scaled, _,_,_,_= minmax_scale_dummy(Xy_train, Xy_test,mask_cat,divide_by=2)
                 # Wasserstein-2 based on L1 cost (after scaling)
                if Xy_train.shape[0] < OTLIM:
                    score_W1_train[method] += pot.emd2(pot.unif(Xy_train_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_train_scaled, Xy_fake_scaled, metric='cityblock')) / (nexp*ngen)
                    score_W1_test[method] += pot.emd2(pot.unif(Xy_test_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_test_scaled, Xy_fake_scaled, metric='cityblock')) / (nexp*ngen)           
                if X.shape[1]==1:
                    X_fake,y_fake = Xy_fake_i[:,0].reshape(-1,1),Xy_fake_i[:,0]
                else:
                    X_fake, y_fake = Xy_fake_i[:,:-1], Xy_fake_i[:,-1]

                #Remove all  NAN values in the test data if there are 
                obs_to_remove_tr, obs_to_remove_te= np.isnan(X_train).any(axis=1), np.isnan(X_test).any(axis=1)
                X_train, X_test,y_train, y_test=X_train[~obs_to_remove_tr], X_test[~obs_to_remove_te],y_train[~obs_to_remove_tr], y_test[~obs_to_remove_te]
                ### Get the predictive performances 
                # Trained on real data 
                f1_real, R2_real = test_on_multiple_models(X_train, y_train, X_test, y_test, mask_cat[:-1],problem_type,  5) # mask_cat[:-1] because we dont need the mask for Y
                # Trained on fake data
                f1_fake, R2_fake = test_on_multiple_models(X_fake, y_fake, X_test, y_test,mask_cat[:-1],problem_type,  5)
                # Trained on real data and fake data
                X_both = np.concatenate((X_train,X_fake), axis=0)
                y_both = np.concatenate((y_train,y_fake))
                f1_both, R2_both = test_on_multiple_models(X_both, y_both, X_test, y_test,mask_cat[:-1],problem_type,  5)
                
                for key in ['mean', 'lin', 'linboost', 'tree', 'treeboost']:
                    f1['real'][key] += f1_real[key] / (nexp*ngen)
                    f1['fake'][key] += f1_fake[key] / (nexp*ngen)
                    f1['both'][key] += f1_both[key] / (nexp*ngen)
                    R2['real'][key] += R2_real[key] / (nexp*ngen)
                    R2['fake'][key] += R2_fake[key] / (nexp*ngen)
                    R2['both'][key] += R2_both[key] / (nexp*ngen)
                # coverage based on L1 cost (after scaling)
                coverage_train[method]  += compute_coverage(Xy_train_scaled, Xy_fake_scaled, None) / (nexp*ngen)
                coverage_test[method]  += compute_coverage(Xy_test_scaled, Xy_fake_scaled, None) / (nexp*ngen)

    #Write results in csv file
    print(f"____The {data_name} data set has been sampled and its performance metrics have been computed_____")    
    # csv_str = f"{data_name}," + method_str + f", {score_W1_train} , {score_W1_test}, {time_taken} " 
    csv_str = f"{data_name}, {method_str}, {score_W1_train[method]} , {score_W1_test[method]} , {R2['real']['mean']} , {R2['fake']['mean']} , {R2['both']['mean']} , {f1['real']['mean']} , {f1['fake']['mean']} ,{f1['both']['mean']} , {coverage_train[method]} , {coverage_test[method]}, {time_taken[method]} " 
    for key in ['lin', 'linboost', 'tree', 'treeboost']:
        csv_str += f",{R2['real'][key]} , {R2['fake'][key]} , {R2['both'][key]} , {f1['real'][key]} , {f1['fake'][key]} , {f1['both'][key]} "
    csv_str += f"\n"
    # ls=["dataset", "method_str",f"score_W1_train[{method}]" ,f" score_W1_test[{method}]", f"time_taken[{method}] "]    
 
    col_ls=["dataset","method_str","W_train" ,"W_test" , "R2_real" , "R2_fake" , "R2_comb", "F1_real","F1_fake","F1_comb" , "coverage_train" ,"coverage_test" ,"time"]   

    result = []
    for key in ['lin', 'linboost', 'tree', 'treeboost']:
        result+=[
            f"R2[{method}]['real'][{key}]",
            f"R2[{method}]['fake'][{key}]",
            f"R2[{method}]['both'][{key}]",
            f"f1[{method}]['real'][{key}]",
            f"f1[{method}]['fake'][{key}]",
            f"f1[{method}]['both'][{key}]"] 
    col_ls=col_ls+result
    #Creating dataframe containing name of these metric used
    m_dt=pd.DataFrame(columns=col_ls)
    csv_st=csv_str.replace("\n","").split(",")
    csv_4_ls=[]
    for i in csv_st:
        try:
            # Try converting to float
            float_value = float(i)
            csv_4_ls.append(float_value)
        except ValueError:
            # If conversion fails, append the original string
            csv_4_ls.append(i)
    m_dt.loc[0]=csv_4_ls
    return m_dt,Xy_fake
