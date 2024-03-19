
import time
import xgboost as xgb
import statsmodels.api as sm
import numpy as np
import pandas as pd
import ot as pot
import sklearn.metrics
import random
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from .Scaling_and_Clipping import Data_processing_functions
from My_package.Sampling_Functions import sampling

def test_on_multiple_models(X_train, y_train, X_test, y_test,cat_indexes,problem_type=None,   nexp=3):
    
    simplefilter("ignore", category=ConvergenceWarning)
    f1_score_lin = 0.0
    f1_score_linboost = 0.0
    f1_score_tree = 0.0
    f1_score_treeboost = 0.0

    R2_score_lin = 0.0
    R2_score_linboost = 0.0
    R2_score_tree = 0.0
    R2_score_treeboost = 0.0

    # Dummify categorical variables (>=3 categories)
#     divide_by=2
    n = X_train.shape[0]
    if len(cat_indexes) > 0:
        X_train_test, _, _,_ = Data_processing_functions.dummify(np.concatenate((X_train, X_test), axis=0), cat_indexes)
        X_train_ = X_train_test[0:n,:]
        X_test_ = X_train_test[n:,:]
    else:
        X_train_ = X_train
        X_test_ = X_test
        

        
    for j in range(nexp):
        if problem_type.capitalize()== "Class":

            if not np.isin(np.unique(y_test), np.unique(y_train)).all(): # not enough classes were generated, score=0
                f1_score_lin += 0
                f1_score_linboost += 0
                f1_score_tree += 0
                f1_score_treeboost += 0
            else:
                # Ensure labels go from 0 to num_classes, otherwise it wont work
                # Assumes uniques(y_train) >= uniques(y_test)
                le = LabelEncoder()
#                 if i=="congress":
                y_train_ = le.fit_transform(y_train)
#                 print(y_train_ )
                y_pred = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500, random_state=j).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
#                 print(y_pred, y_test)  
                f1_score_lin += f1_score(y_test, y_pred, average='macro') / nexp

                y_pred = AdaBoostClassifier(random_state=j).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_linboost += f1_score(y_test, y_pred, average='macro') / nexp

                y_pred = RandomForestClassifier(max_depth=28, random_state=j).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_tree += f1_score(y_test, y_pred, average='macro') / nexp

                y_pred = xgb.XGBClassifier(reg_lambda=0.0, random_state=j).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_treeboost += f1_score(y_test, y_pred, average='macro') / nexp
        elif problem_type.capitalize()== "Reg":
            y_pred = LinearRegression().fit(X_train_, y_train).predict(X_test_)
            R2_score_lin += r2_score(y_test, y_pred) / nexp

            y_pred = AdaBoostRegressor(random_state=j).fit(X_train_, y_train).predict(X_test_)
            R2_score_linboost += r2_score(y_test, y_pred) / nexp

            y_pred = RandomForestRegressor(max_depth=28, random_state=j).fit(X_train_, y_train).predict(X_test_)
            R2_score_tree += r2_score(y_test, y_pred) / nexp

            y_pred = xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0.0, random_state=j).fit(X_train_, y_train).predict(X_test_)
            R2_score_treeboost += r2_score(y_test, y_pred) / nexp
        else: 
            raise Exception("The performance metrics are made for for regression and classification problem only, please choose the right choice for argument <problem_type>")
    f1_score_mean = (f1_score_lin + f1_score_linboost + f1_score_tree + f1_score_treeboost) / 4
    R2_score_mean = (R2_score_lin + R2_score_linboost + R2_score_tree + R2_score_treeboost) / 4

    return {'mean': f1_score_mean, 'lin': f1_score_lin, 'linboost': f1_score_linboost, 'tree': f1_score_tree, 'treeboost': f1_score_treeboost}, {'mean': R2_score_mean, 'lin': R2_score_lin, 'linboost': R2_score_linboost, 'tree': R2_score_tree, 'treeboost': R2_score_treeboost}

def compute_pairwise_distance(data_x, data_y=None): # Changed to L1 instead of L2 to better handle mixed data
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='cityblock', n_jobs=-1)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


# Automatically finding the best k as per https://openreview.net/pdf?id=1mNssCWt_v
def compute_coverage(real_features, fake_features, nearest_k=None):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    if nearest_k is None: # we choose k to be the smallest such that we have 95% coverage with real data
        coverage_ = 0
        nearest_k = 1
        while coverage_ < 0.95:
            coverage_ = compute_coverage(real_features, real_features, nearest_k)
            nearest_k += 1

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return coverage

# print(n)
# Need to train/test split for evaluating the linear regression performance and for W2 based on test
def define_data_class_or_regr(X,y,n):
    stratify_option = y if Data_processing_functions.all_integers(y) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n, stratify=stratify_option)
    if len(X.shape)==1:
       
        X_train = X_train.reshape(-1,1)
        X_test=X_test.reshape(-1,1)
        Xy_train = np.copy(X_train)
        Xy_test=np.copy(X_test)
    else:
        Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
        Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)
    return Xy_train, Xy_test,X_train, X_test, y_train, y_test

def Metrics(ngen,nexp,diffusion_model,dt_loader,dt_name,
            N,K_dpl,which_solver,model_type,Use_OneHotEnc,cat_sampler_type,arg1,arg2,problem_type=None,method=None,forest_flow=None,mask_cat=None):
    """ ngen and nexp: Number of generation and experiment
        diffusion_model: the diffusion function you have to input ( the parameters should be dt_loader,i,N and K_dpl)
        dt_loader and dt_name: are respectively the data  and the name of the data you imputted
        problem_type: takes two arguments {Regression: if the data was studied for a regression or Classification: if it was for a classification}
        model_type: chooses whether we opt for regressor only or regressor mixed with classifier
        N,K_dpl,which_solver,method: are the number of noise levels, the number of duplication, a string that provide the name of the solver we want the options are (Euler: for euler solver, Rg4: for: Runge Kutta 4th order, or Any_string: for both).
        mask_cat: is the list that shows wether or not the feature of your data set are categorical 
        Use_OneHotEnc: Determine whether or not we will use one hot encoding (takes argument True or False)
    """
    # dt_name=dt_name.split("__")[0]
    OTLIM = 5000
    if forest_flow==None:
        method="VSFF"
    else:
        method="Forest Flow"
            
    score_W1_train = {}
    score_W1_test = {}
    coverage = {}
    coverage_test = {}
    time_taken = {}
    percent_bias = {}
    coverage_rate = {}
    AW = {}
    # for method in args.methods:
    method_str =method
    score_W1_test[method] = 0.0
    score_W1_train[method] = 0.0
    coverage[method] = 0.0
    coverage_test[method] = 0.0
    time_taken[method] = 0.0
    percent_bias[method] = 0.0
    coverage_rate[method] = 0.0
    AW[method] = 0.0
    R2 = {}
    f1 = {}
    # for method in args.methods:
    R2[method] = {'real': {}, 'fake': {}, 'both': {}}
    f1[method] = {'real': {}, 'fake': {}, 'both': {}}
    for test_type in ['real','fake','both']:
        for test_type2 in ['mean','lin','linboost', 'tree', 'treeboost']:
            R2[method][test_type][test_type2] = 0.0
            f1[method][test_type][test_type2] = 0.0
    if mask_cat== None:
        mask_cat=dt_loader[-1] #This is the mask list that represent all the column that are categorical and those that are not
    
    cat_indexes=[i for i in range(len(mask_cat)) if mask_cat[i]]
    obs_to_remove = np.isnan(dt_loader[0]).all(axis=1)
    dta = dt_loader[0][~obs_to_remove]
    if dta.shape[1]==1:
        X=y=dt_loader[0][:,0]
       
    else:
        X,y=dta[:,:-1],dta[:,-1]
    b,c=dt_loader[0].shape
   

    for n in range(nexp):
            Xy_train, Xy_test,X_train, X_test, y_train, y_test=define_data_class_or_regr(X,y,n)
            start = time.time()   
            # Determining  a tensor of ngen generated samples   
            if forest_flow== None:
                Xy_fake=np.array([diffusion_model(dt_loader,N,K_dpl,model_type,Use_OneHotEnc,cat_sampler_type,which_solver,arg1,arg2).sample() for k in range(ngen)])
            else:
                Xy_fake= np.array([diffusion_model(dt_loader,N,K_dpl) for k in range(ngen)])
            end = time.time()
            time_taken[method] += (end - start) / nexp
            for gen_i in range(ngen):

                Xy_fake_i=Xy_fake[gen_i]
                # print(Xy_fake_i.shape)
                Xy_fake_i_train = Xy_fake[gen_i][:Xy_train.shape[0]]
                Xy_fake_i_test = Xy_fake[gen_i][Xy_train.shape[0]:]
                Xy_train_scaled, Xy_fake_scaled,_,_,_,_= Data_processing_functions.minmax_scale_dummy(Xy_train, Xy_fake_i, mask_cat)
                _, Xy_test_scaled, _,_,_,_= Data_processing_functions.minmax_scale_dummy(Xy_train, Xy_test,  mask_cat)
#                 print("Xy_train.shape,Xy_fake_scaled.shape,Xy_test_scaled.shape:",(Xy_train_scaled.shape,Xy_fake_scaled.shape,Xy_test_scaled.shape))

                # Wasserstein-2 based on L1 cost (after scaling)
                if Xy_train.shape[0] < OTLIM:
                    score_W1_train[method] += pot.emd2(pot.unif(Xy_train_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_train_scaled, Xy_fake_scaled, metric='cityblock')) / (nexp*ngen)
                    score_W1_test[method] += pot.emd2(pot.unif(Xy_test_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_test_scaled, Xy_fake_scaled, metric='cityblock')) / (nexp*ngen)

            
               #
                if dt_loader[0].shape[1]==1:
                    X_fake,y_fake = Xy_fake_i[:,0].reshape(-1,1),Xy_fake_i[:,0]
                else:
                    X_fake, y_fake = Xy_fake_i[:,:-1], Xy_fake_i[:,-1]

                # Trained on real data
                f1_real, R2_real = test_on_multiple_models(X_train, y_train, X_test, y_test, mask_cat[:-1],problem_type,  nexp)

                # Trained on fake data
                f1_fake, R2_fake = test_on_multiple_models(X_fake, y_fake, X_test, y_test,mask_cat[:-1],problem_type,  nexp)
# 
                # Trained on real data and fake data
                X_both = np.concatenate((X_train,X_fake), axis=0)
                y_both = np.concatenate((y_train,y_fake))
                f1_both, R2_both = test_on_multiple_models(X_both, y_both, X_test, y_test,mask_cat[:-1],problem_type,  nexp)
                
                for key in ['mean', 'lin', 'linboost', 'tree', 'treeboost']:
                    f1[method]['real'][key] += f1_real[key] / (nexp*ngen)
                    f1[method]['fake'][key] += f1_fake[key] / (nexp*ngen)
                    f1[method]['both'][key] += f1_both[key] / (nexp*ngen)
                    R2[method]['real'][key] += R2_real[key] / (nexp*ngen)
                    R2[method]['fake'][key] += R2_fake[key] / (nexp*ngen)
                    R2[method]['both'][key] += R2_both[key] / (nexp*ngen)

                # coverage based on L1 cost (after scaling)
                coverage[method] += compute_coverage(Xy_train_scaled, Xy_fake_scaled, None) / (nexp*ngen)
                coverage_test[method] += compute_coverage(Xy_test_scaled, Xy_fake_scaled, None) / (nexp*ngen)

#         Write results in csv file
    print(f"____The {dt_name} data set has been sampled and its performance metrics have been computed_____")    
    csv_str = f"{dt_name} , " + method_str + f", {score_W1_train[method]} , {score_W1_test[method]} , {R2[method]['real']['mean']} , {R2[method]['fake']['mean']} , {R2[method]['both']['mean']} , {f1[method]['real']['mean']} , {f1[method]['fake']['mean']} ,{f1[method]['both']['mean']} , {coverage[method]} , {coverage_test[method]}  , {time_taken[method]} " 
    for key in ['lin', 'linboost', 'tree', 'treeboost']:
        csv_str += f",{R2[method]['real'][key]} , {R2[method]['fake'][key]} , {R2[method]['both'][key]} , {f1[method]['real'][key]} , {f1[method]['fake'][key]} , {f1[method]['both'][key]} "
    csv_str += f"\n"


    ls=["dataset", "method_str", f"score_W1_train[{method}]" ,f" score_W1_test[{method}]" , f"R2[{method}]['real']['mean']" , f"R2[{method}]['fake']['mean']" , f"R2[{method}]['both']['mean']" , f"f1[{method}]['real']['mean']" , f"f1[{method}]['fake']['mean']" , f"f1[{method}]['both']['mean']" , f"coverage[{method}]" ,f"coverage_test[{method}]" ,f"time_taken[{method}] "]
    #print(csv_str)
    
    result = []
    for key in ['lin', 'linboost', 'tree', 'treeboost']:
        result+=[
            f"R2[{method}]['real'][{key}]",
            f"R2[{method}]['fake'][{key}]",
            f"R2[{method}]['both'][{key}]",
            f"f1[{method}]['real'][{key}]",
            f"f1[{method}]['fake'][{key}]",
            f"f1[{method}]['both'][{key}]"]
    #     
    ls=ls+result
#     print(len(ls))
    #Creating dataframe containing name of the metric used
    m_dt=pd.DataFrame(columns=ls)
    #

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
