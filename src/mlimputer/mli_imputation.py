import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import atlantic as atl
from .mli_parameters import imputer_parameters
from .mli_model_selection import missing_report, imput_models

parameters=imputer_parameters()

def fit_imput(Dataset:pd.DataFrame,
              imput_model:str,
              imputer_configs:dict=parameters):
    
    """
    This function fits missing data in a pandas dataframe using the imputation method specified by the user.
    
    Parameters:
    Dataset (pd.DataFrame): The input pandas dataframe that needs to be imputed.
    imput_model (str, optional): The imputation method to be used. Default is "KNN".
    
    Returns:
    imp_config (dict): A dictionary containing the fitted imputation models for each column with missing data.
    """      
    
    df=Dataset.copy()

    df_md,c=missing_report(df),0
    imp_targets=list(df_md['columns']) 
        
    # Iterate over each column with missing data and fit the imputation method
    for col in tqdm(imp_targets, desc="Fitting Missing Data Columns", ncols=80): ## imp_targets:
        #print("**** Fitting Column:", col)
        Target=col
        
        # Split the data into train and test sets
        total_index = df.index.tolist()
        test_index = df[df[Target].isnull()].index.tolist()
        train_index = [value for value in total_index if value not in test_index]
        
        Train=df.iloc[train_index]
        
        # Fit the label encoding method in categorical columns
        le_fit=atl.fit_Label_Encoding(Train,Target)
        Train=atl.transform_Label_Encoding(Train,le_fit)
        
        # Fit the simple imputation method in input columns
        imputer_simple=atl.fit_SimpleImp(df=Train,
                                         target=Target,
                                         strat='mean')
        
        Train=atl.transform_SimpleImp(df=Train,
                                      target=Target,
                                      imputer=imputer_simple)
        # Fit the imputation model
        model = imput_models(Train=Train,
                             target=Target,
                             parameters=imputer_configs,
                             algo=imput_model)
        
        # Store the fitted model information in a dictionary
        if c==0:
            imp_config = {Target:{'model_name':imput_model,
                                  'model':model,
                                  'pre_process':le_fit,
                                  'input_nulls':imputer_simple}}
        elif c>0:
            imp_config_2 = {Target:{'model_name':imput_model,
                                    'model':model,
                                    'pre_process':le_fit,
                                    'input_nulls':imputer_simple}}
            imp_config.update(imp_config_2)
        c+=1
        
    return imp_config

def transform_imput(Dataset:pd.DataFrame,
                    fit_configs:dict):
    """
    Imputation of missing values in a dataset using a pre-fit imputation model.
    
    Parameters:
    -----------
    Dataset: pd.DataFrame
        The dataset containing missing values to be imputed.
    fit_configs: dict
        A dictionary of pre-fit imputation models and associated pre-processing information
        for each column with missing values in the dataset.
        
    Returns:
    --------
    df_: pd.DataFrame
        The original dataset with missing values imputed.
    """
    df_,imp_cols=Dataset.copy(),list(fit_configs.keys()) #[0]
    
    for col in tqdm(imp_cols, desc="Imputing Missing Data", ncols=80):#in imp_cols:
        
        Target=col
        test_index = df_[df_[Target].isnull()].index.tolist()
        test_df=df_.iloc[test_index]
        
        le_fit=fit_configs[Target]['pre_process']
        test_df=atl.transform_Label_Encoding(test_df,le_fit)
        input_num_cols = atl.num_cols(test_df, Target)
        
        imputer_simple=fit_configs[Target]['input_nulls']
        test_df=atl.transform_SimpleImp(df=test_df,
                                        target=Target,
                                        imputer=imputer_simple)
        
        sel_cols=list(test_df.columns)
        sel_cols.remove(Target)
        sel_cols.append(Target)
        test_df=test_df[sel_cols]
        X_test = test_df.iloc[:, 0:(len(sel_cols)-1)].values

        model=fit_configs[Target]['model']
    
        y_predict = model.predict(X_test)

        df_[Target].iloc[test_index]=y_predict

    return df_