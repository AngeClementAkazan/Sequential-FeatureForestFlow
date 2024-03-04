# data=dataset="iris"
from sklearn.exceptions import ConvergenceWarning
import time
import numpy as np
import pandas as pd
import ot as pot
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from My_package.Scaling_and_Clipping import Data_processing_functions
from My_package.Sampling_Functions import sampling
from tensorflow.python.ops.numpy_ops import np_config

# from utils import dummify

import copy

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


def test_on_multiple_models(X_train, y_train, X_test, y_test,dt_set_name,i,cat_indexes,classifier=None,   nexp=3):

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
        
    if i in dt_set_name[:-1]:
        classifier=True
    elif i in dt_set_name[-1]:
        classifier=False
        
    for j in range(nexp):
        if classifier:
#             if i=="congress":
#                 print(np.isin(np.unique(y_test), np.unique(y_train)).all())
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
        else:
            y_pred = LinearRegression().fit(X_train_, y_train).predict(X_test_)
            R2_score_lin += r2_score(y_test, y_pred) / nexp

            y_pred = AdaBoostRegressor(random_state=j).fit(X_train_, y_train).predict(X_test_)
            R2_score_linboost += r2_score(y_test, y_pred) / nexp

            y_pred = RandomForestRegressor(max_depth=28, random_state=j).fit(X_train_, y_train).predict(X_test_)
            R2_score_tree += r2_score(y_test, y_pred) / nexp

            y_pred = xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=0.0, random_state=j).fit(X_train_, y_train).predict(X_test_)
            R2_score_treeboost += r2_score(y_test, y_pred) / nexp

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
    Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)
    return Xy_train, Xy_test,X_train, X_test, y_train, y_test

def Metrics(ngen,nexp,diffusion_model,dt_loader,dt_set_name,i,N,K_dpl,Which_solver,method=None,forest_flow=None,mask_cat=None):
    """ ngen and nexp: Number of generation and experiment
        diffusion_model: the diffusion function you have to input ( the parameters should be dt_loader,i,N and K_dpl)
        dt_loader,dt_set_name and i: are respectively the data you input, the list for the data set names and the iterator on that list.
        N,K_dpl,Which_solver,method: are the number of noise levels, the number of duplication, a string that provide the name of the solver we want the options are (Euler: for euler solver, Rg4: for: Runge Kutta 4th order, or Any_string: for both).
        mask_cat: is the list that shows wether or not the feature of your data set are categorical
        """
        
    OTLIM = 5000
    if forest_flow==False:
        method="VSFF"
    else:
        method="Forest Flow"
            
    score_W2_train = {}
    score_W2_test = {}
    coverage = {}
    coverage_test = {}
    time_taken = {}
    percent_bias = {}
    coverage_rate = {}
    AW = {}
    # for method in args.methods:
    method_str =method
    score_W2_test[method] = 0.0
    score_W2_train[method] = 0.0
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
#             sample(data_loader,data_set_name,euler_solve,runge_kutta_solve,10,50)
    mask_cat=dt_loader(i)[-1]  #This is the mask list that represent all the column that are categorical and those that are not
    data=dt_loader(i)[0]
    cat_indexes=[i for i in range(len(mask_cat)) if mask_cat[i]]
    X,y=dt_loader(i)[0][:,:-1],dt_loader(i)[0][:,-1]
#     b,c=data.shape
    for n in range(nexp):
            Xy_train, Xy_test,X_train, X_test, y_train, y_test=define_data_class_or_regr(X,y,n)
#             print()
            start = time.time()

            if forest_flow==False:
            
                smp=diffusion_model(dt_loader,i,N,K_dpl,Which_solver)
                solution=smp.sample()
                Xy_fake= np.array([solution for k in range(ngen)])
            else:
                Xy_fake= np.array([diffusion_model(dt_loader,i,N,K_dpl) for k in range(ngen)])
            end = time.time()
            time_taken[method] += (end - start) / nexp
            for gen_i in range(ngen):
                Xy_fake_i = Xy_fake[gen_i]

                Xy_train_scaled, Xy_fake_scaled,_,_,_,_= Data_processing_functions.minmax_scale_dummy(Xy_train, Xy_fake_i, mask_cat,divide_by=2)
                _, Xy_test_scaled, _,_,_,_= Data_processing_functions.minmax_scale_dummy(Xy_train, Xy_test,  mask_cat,divide_by=2)
               
                # Wasserstein-2 based on L1 cost (after scaling)
                if Xy_train.shape[0] < OTLIM:
                    score_W2_train[method] += np.sqrt(pot.emd2(pot.unif(Xy_train_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_train_scaled, Xy_fake_scaled, metric='cityblock'))) / (nexp*ngen)
                    score_W2_test[method] += np.sqrt(pot.emd2(pot.unif(Xy_test_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_test_scaled, Xy_fake_scaled, metric='cityblock'))) / (nexp*ngen)

                X_fake, y_fake = Xy_fake_i[:,:-1], Xy_fake_i[:,-1]

                # Trained on real data
                f1_real, R2_real = test_on_multiple_models(X_train, y_train, X_test, y_test,dt_set_name,i, cat_indexes,  nexp)

                # Trained on fake data
                f1_fake, R2_fake = test_on_multiple_models(X_fake, y_fake, X_test, y_test,dt_set_name,i, cat_indexes,  nexp)
#                 print(f1_fake)
                # Trained on real data and fake data
                X_both = np.concatenate((X_train,X_fake), axis=0)
                y_both = np.concatenate((y_train,y_fake))
                f1_both, R2_both = test_on_multiple_models(X_both, y_both, X_test, y_test,dt_set_name,i, cat_indexes,  nexp)
                
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
    print(f"The {i} data set has been successfully sampled, thank you to use my sampler")    
    csv_str = f"{i} , " + method_str + f", {score_W2_train[method]} , {score_W2_test[method]} , {R2[method]['real']['mean']} , {R2[method]['fake']['mean']} , {R2[method]['both']['mean']} , {f1[method]['real']['mean']} , {f1[method]['fake']['mean']} ,{f1[method]['both']['mean']} , {coverage[method]} , {coverage_test[method]}  , {time_taken[method]} " 
    for key in ['lin', 'linboost', 'tree', 'treeboost']:
        csv_str += f",{R2[method]['real'][key]} , {R2[method]['fake'][key]} , {R2[method]['both'][key]} , {f1[method]['real'][key]} , {f1[method]['fake'][key]} , {f1[method]['both'][key]} "
    csv_str += f"\n"


    ls=["dataset", "method_str", f"score_W2_train[{method}]" ,f" score_W2_test[{method}]" , f"R2[{method}]['real']['mean']" , f"R2[{method}]['fake']['mean']" , f"R2[{method}]['both']['mean']" , f"f1[{method}]['real']['mean']" , f"f1[{method}]['fake']['mean']" , f"f1[{method}]['both']['mean']" , f"coverage[{method}]" ,f"coverage_test[{method}]" ,f"time_taken[{method}] "]
    #     print(csv_str)
    
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
#     method_index_start = 0 #  so we loop back againsample_Euler(X,y, euler_solve,runge_kutta_solve,b,c,n_t)Metric,Sample=