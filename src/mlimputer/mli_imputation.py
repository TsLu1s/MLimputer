import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import atlantic as atl
from .mli_parameters import imputer_parameters
from .mli_model_selection import missing_report, imput_models

parameters=imputer_parameters()

def fit_imput(dataset:pd.DataFrame,
              imput_model:str,
              imputer_configs:dict=parameters):
    
    """
    This function fits missing data in a pandas dataframe using the imputation method specified by the user.
    
    Parameters:
    dataset (pd.DataFrame): The input pandas dataframe that needs to be imputed.
    imput_model (str, optional): The imputation method to be used. Default is "KNN".
    
    Returns:
    imp_config (dict): A dictionary containing the fitted imputation models for each column with missing data.
    """      
    
    df=dataset.copy()

    df_md,c=missing_report(df),0
    imp_targets=list(df_md['columns']) 
        
    for col in df.columns:
        if df[col].isnull().all():
            raise ValueError(f'Column {col} is filled with null values')
    
    # Iterate over each column with missing data and fit the imputation method
    for col in tqdm(imp_targets, desc="Fitting Missing Data Columns", ncols=80): ## imp_targets:
        #print("**** Fitting Column:", col)
        target=col
        
        # Split the data into train and test sets
        total_index = df.index.tolist()
        test_index = df[df[target].isnull()].index.tolist()
        train_index = [value for value in total_index if value not in test_index]
        
        train=df.iloc[train_index]
        
        # Fit the label encoding method in categorical columns
        le_fit=atl.fit_Label_Encoding(train,target)
        train=atl.transform_Label_Encoding(train,le_fit)
        
        # Fit the simple imputation method in input columns
        imputer_simple=atl.fit_SimpleImp(dataset=train,
                                         target=target,
                                         strat='mean')
        
        train=atl.transform_SimpleImp(dataset=train,
                                      target=target,
                                      imputer=imputer_simple)
        # Fit the imputation model
        model = imput_models(train=train,
                             target=target,
                             parameters=imputer_configs,
                             algo=imput_model)
        
        # Store the fitted model information in a dictionary
        if c==0:
            imp_config = {target:{'model_name':imput_model,
                                  'model':model,
                                  'pre_process':le_fit,
                                  'input_nulls':imputer_simple}}
        elif c>0:
            imp_config_2 = {target:{'model_name':imput_model,
                                    'model':model,
                                    'pre_process':le_fit,
                                    'input_nulls':imputer_simple}}
            imp_config.update(imp_config_2)
        c+=1
        
    return imp_config

def transform_imput(dataset:pd.DataFrame,
                    fit_configs:dict):
    """
    Imputation of missing values in a dataset using a pre-fit imputation model.
    
    Parameters:
    -----------
    dataset: pd.DataFrame
        The dataset containing missing values to be imputed.
    fit_configs: dict
        A dictionary of pre-fit imputation models and associated pre-processing information
        for each column with missing values in the dataset.
        
    Returns:
    --------
    df_: pd.DataFrame
        The original dataset with missing values imputed.
    """
    df_,imp_cols=dataset.copy(),list(fit_configs.keys()) #[0]
    
    for col in tqdm(imp_cols, desc="Imputing Missing Data", ncols=80):#in imp_cols:
        
        target=col
        test_index = df_[df_[target].isnull()].index.tolist()
        test_df=df_.iloc[test_index]
        
        le_fit=fit_configs[target]['pre_process']
        test_df=atl.transform_Label_Encoding(test_df,le_fit)
        input_num_cols = atl.num_cols(test_df, target)
        
        imputer_simple=fit_configs[target]['input_nulls']
        test_df=atl.transform_SimpleImp(dataset=test_df,
                                        target=target,
                                        imputer=imputer_simple)
        
        sel_cols=list(test_df.columns)
        sel_cols.remove(target)
        sel_cols.append(target)
        test_df=test_df[sel_cols]
        X_test = test_df.iloc[:, 0:(len(sel_cols)-1)].values

        model=fit_configs[target]['model']
    
        y_predict = model.predict(X_test)

        df_[target].iloc[test_index]=y_predict

    return df_
