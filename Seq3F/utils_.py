import copy
import xgboost as xgb
import sklearn.metrics
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

### This file contains the functions used to evaluate models our generated data using R2 F1 and coverage, it is mainly origionated from the Forest ###
def test_on_multiple_models(X_train, y_train, X_test, y_test,cat_indexes,problem_type=None,   nexp=5):    
    simplefilter("ignore", category=ConvergenceWarning)
    f1_score_lin = 0.0
    f1_score_linboost = 0.0
    f1_score_tree = 0.0
    f1_score_treeboost = 0.0
    R2_score_lin = 0.0
    R2_score_linboost = 0.0
    R2_score_tree = 0.0
    R2_score_treeboost = 0.0
    n = X_train.shape[0]
    if len(cat_indexes) > 0:
        X_train_test, _, _,_ = dummify(np.concatenate((X_train, X_test), axis=0), cat_indexes)
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
                y_pred = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500, random_state=j).fit(X_train_, y_train_).predict(X_test_)
                y_pred = le.inverse_transform(y_pred)
                f1_score_lin += f1_score(y_test, y_pred, average='macro') / nexp

                y_pred = AdaBoostClassifier(random_state=j,algorithm='SAMME').fit(X_train_, y_train_).predict(X_test_)
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
            raise Exception("The performance metrics are made for regression and classification problem only, please choose the right choice for argument <problem_type>")
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

def dummify_4_metrics(X, cat_indexes, divide_by=2, drop_first=False): #To use in the metrics scripts in order to foster wasserstein computation and result comparisons
        df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
        df_names_before = df.columns
        for i in range(len(cat_indexes)):
            if cat_indexes[i]==True:
                df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=drop_first)
                if divide_by > 0: # needed for L1 distance to equal 1 when categories are different
                    filter_col = [col for col in df if col.startswith(str(i) + '_')]
                    df[filter_col] = df[filter_col] / divide_by
        df_names_after = df.columns 
        df = df.to_numpy()
        cat_index=[]
        for j in df_names_after:
            if "_" in j:  # Check if the last added column has "_"
                cat_index.append(True)
            else:
                cat_index.append(False)
        return df, df_names_before, df_names_after,cat_index

def dummify(X,mask,  drop_first=False):
        df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])  # Convert to Pandas       
        df_names_before = df.columns
        mask_ = []  # List to store categorical column indexes
        for i in range(len(mask)):
            if mask[i]==True:
                df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=drop_first)
        for j in df.columns:
            if "_" in j:  # Check if the last added column has "_"
                mask_.append(True)
            else:
                mask_.append(False)
        df_names_after = df.columns
        df = df.to_numpy()
        return df, df_names_before, df_names_after, mask_
# def calculate_vif(df,mask):
#     if mask[:-1].count(True)>1:
#         df,nbf,naft,m=dummify(df[:-1],mask,  drop_first=True)
#     df = pd.DataFrame(df, columns=[str(i) for i in naft])  
#     vif_data = pd.DataFrame()
#     vif_data['Feature'] = df.columns
#     vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
#     return vif_data['VIF'].mean(numeric_only=True)  

def minmax_scale_dummy(X_train, X_test, msk_ct,divide_by=2, mask=None):
    X_train_ = copy.deepcopy(X_train)
    X_test_ = copy.deepcopy(X_test)
    scaler = MinMaxScaler()
    num_cat=msk_ct.count(True)
    if num_cat!= X_train_.shape[1]: # if some variables are continuous, we  scale-transform
        not_cat_indexes=~np.array(msk_ct)  #Full 1D array indicating non categorical as True and categorical as False
        scaler.fit(X_train_[:, not_cat_indexes])
        X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
        X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])
    # One-hot the categorical variables 
    df_names_before, df_names_after = None, None
    n = X_train.shape[0]  
    X_train_t = X_train_
    X_test_t= X_test_
    if num_cat > 0:                                
        X_train_test, df_names_before, df_names_after,mask=dummify_4_metrics(np.concatenate((X_train_, X_test_), axis=0),msk_ct,divide_by, drop_first=False)
        X_train_t = X_train_test[0:n,:]
        X_test_t= X_train_test[n:,:]
    return X_train_t, X_test_t, scaler,mask, df_names_before, df_names_after

# print(f"Column {col}: {len(unique_values)} unique categories, correctly in the range [0, ..., {N}]")  
# Seperate dataset into multiple minibatches for memory-efficient training 
# Check the category type 
def check_categorical_columns(X, categorical_columns):
      for col in categorical_columns:
          unique_values = np.unique(X[:, col])  # Find unique values in the column
          N = unique_values.max()  # Maximum category value
          expected_values = np.arange(0, N + 1)  # Expected range [0, ..., N]
          assert np.array_equal(unique_values, expected_values), \
              f"Column {col} has missing categories or values outside the range [0, ..., {N}]. Use label encoding if necessary."

class IterForDMatrix(xgb.core.DataIter):
    """A data iterator for XGBoost DMatrix.
    `reset` and `next` are required for any data iterator, the other functions here
    are utilites for demonstration's purpose.
    mask_cat and and get_xt_y are respectively functions to concatenate the input of the velocity vector and get a conditional flow at time t
    """
    def __init__(self, data,mask_cat,n_t,get_xt_y,make_mat,t,i, dim,model_type="HS3F",   n_batch=1000, n_epochs=10, eps=1e-3):
        self._data = data
        self.n_batch = n_batch
        self.n_epochs = n_epochs
        self.t = t
        self.make_mat=make_mat
        self.eps = eps
        self.dim = dim
        self.it = 0  # set iterator to 0
        self.n_t=n_t
        self.i=i
        self.model_type=model_type
        self.mask_cat=mask_cat
        self.get_xt_y=get_xt_y
        super().__init__()

    def reset(self):
      """Reset the iterator"""
      self.it = 0

    def next(self, input_data):
      """Yield next batch of data."""
      if self.it == self.n_batch*self.n_epochs: # stops after k epochs
        return 0
      x1=self._data[self.it % self.n_batch]
      if self.dim==0:
        if not self.mask_cat[self.dim]: 
          X_t,y= self.get_xt_y(x1,self.dim,self.t,self.i) 
          y_no_miss = ~np.isnan(y.ravel())
          input_data(data=X_t[y_no_miss, :], label=y[y_no_miss])
          self.it += 1  
        #If this is not respected then we move because the first categorical variable is generated using a multinomial sampling based on the class frequencies of this variable 
      else:
        if self.mask_cat[self.dim] and self.model_type== "HS3F": 
          x_t,y= self.make_mat(x1,self.dim),x1[:,self.dim]   
        else:
          X_t,y= self.get_xt_y(x1,self.dim,self.t,self.i) 
          x_t= self.make_mat(x1,self.dim,X_t) 
        y_no_miss = ~np.isnan(y.ravel())
        input_data(data=x_t[y_no_miss, :], label=y[y_no_miss])
        self.it += 1
      return 1