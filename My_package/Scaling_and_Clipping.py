import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler


" This script contains data processing functions"
class Data_processing_functions:

    @staticmethod
    def dummify(X,mask,  drop_first=True):
        df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])  # Convert to Pandas
        df_names_before = df.columns

        cat_indexes = []  # List to store categorical column indexes

        for i in range(len(mask)):
            if mask[i]==True:
                df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=drop_first)
        for j in df.columns:
            if "_" in j:  # Check if the last added column has "_"
                cat_indexes.append(True)
            else:
                cat_indexes.append(False)
        df_names_after = df.columns
        df = df.to_numpy()
        return df, df_names_before, df_names_after, cat_indexes
    
    @staticmethod
    def all_integers(column):
        # Check if all elements are integers
        are_integers = np.equal(column, np.round(column))

        # Check if the decimal part of all elements is zero
        decimal_part_is_zero = np.equal(np.modf(column)[0], 0)

        # Combine the two conditions
        result = np.logical_and(are_integers, decimal_part_is_zero)

        return result.all()
    @staticmethod
    def Dummify(X, cat_indexes, divide_by, drop_first=False):
        df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
        df_names_before = df.columns
        for i in cat_indexes:
            df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=drop_first)
            if divide_by > 0: # needed for L1 distance to equal 1 when categories are different
                filter_col = [col for col in df if col.startswith(str(i) + '_')]
                df[filter_col] = df[filter_col] / divide_by
        df_names_after = df.columns
        df = df.to_numpy()
        cat_index=Data_processing_functions.dummify(X,cat_indexes,  drop_first=True)[-1]
        return df, df_names_before, df_names_after,cat_index

    @staticmethod
    def minmax_scale_dummy(X_train, X_test, msk_ct,divide_by, mask=None):
        X_train_ = copy.deepcopy(X_train)
        X_test_ = copy.deepcopy(X_test)
        scaler = MinMaxScaler()
        cat_ind_only=[i for i in range(len(msk_ct)) if msk_ct[i]]
        
        if len(cat_ind_only) != X_train_.shape[1]: # if some variables are continuous, we  scale-transform
            not_cat_indexes=~np.array(msk_ct)  #Full 1D array indicating non categorical as True and categorical as False
            
            scaler.fit(X_train_[:, not_cat_indexes])
            X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
            X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])
        # One-hot the categorical variables 
        df_names_before, df_names_after = None, None
        n = X_train.shape[0]
        if len(cat_ind_only) > 0:                                
            X_train_test, df_names_before, df_names_after,mask= Data_processing_functions.Dummify(np.concatenate((X_train_, X_test_), axis=0),cat_ind_only,divide_by, drop_first=False)
            X_train_ = X_train_test[0:n,:]
            X_test_ = X_train_test[n:,:]
        else:
            X_train_ = X_train_test[0:n,:]
            X_test_ = X_train_test[n:,:]

        return X_train_, X_test_, scaler,mask, df_names_before, df_names_after

    "Rounding for the categorical variables which are dummy-coded and then remove dummy-coding"
    @staticmethod
    def clean_onehot_data(X,X_names_before,X_names_after,ct_indexes): 
        cat_indexes=[ii for ii in range(len(ct_indexes)) if ct_indexes[ii]]
        if len(cat_indexes) > 0: # ex: [5, 3] and X_names_after [gender_a gender_b cartype_a cartype_b cartype_c]
            
            X_names_after = copy.deepcopy(X_names_after.to_numpy())
            prefixes = [x.split('_')[0] for x in X_names_after if '_' in x] # for all categorical variables, we have prefix ex: ['gender', 'gender']

            unique_prefixes = sorted(set(prefixes ), key=lambda x: int(x))# uniques prefixes
            
            for i in range(len(unique_prefixes)):
                cat_vars_indexes = [unique_prefixes[i] + '_' in my_name for my_name in X_names_after]
                cat_vars_indexes = np.where(cat_vars_indexes)[0] # actual indexes

                cat_vars = X[:, cat_vars_indexes] # [b, c_cat]
                
                # dummy variable, so third category is true if all dummies are 0
                cat_vars = np.concatenate((np.ones((cat_vars.shape[0], 1))*0.5,cat_vars), axis=1)
                
                # argmax of -1, -1, 0 is 0; so as long as they are below 0 we choose the implicit-final class
                max_index = np.argmax(cat_vars, axis=1) # argmax across all the one-hot features (most likely category)
                
                X[:, cat_vars_indexes[0]] = max_index
                X_names_after[cat_vars_indexes[0]] = unique_prefixes[i] # gender_a -> gender
            df = pd.DataFrame(X, columns = X_names_after) # to Pandas
            
            df = df[X_names_before] # remove all gender_b, gender_c and put everything in the right order
            X = df.to_numpy()
        return X
    
#     @staticmethod       

    
    @staticmethod 
    def clipping(min,max,sol,dt_loader,msk_cat):
#         for o in range(dt_loader.shape[1]):
#             if np.all(np.equal(dt_loader[:,o], dt_loader[:,o].astype(int))) or  Data_processing_functions.all_integers(dt_loader[:,o]):
#                 sol[:,o] = np.round(sol[:,o], decimals=0)
        small = (sol < min).astype(float)
        sol= small*min + (1-small)*sol
        big = (sol> max).astype(float)
        sol = big*max + (1-big)*sol
        return sol