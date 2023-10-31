import pandas as pd
from tqdm import tqdm
from atlantic.processing import AutoLabelEncoder
from atlantic.imputation import AutoSimpleImputer
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
        
        cat_cols=[col for col in train.select_dtypes(include=['object']).columns if col != target]
        
        ## Create Label Encoder
        encoder = AutoLabelEncoder()
        ## Fit Label Encoder
        encoder.fit(train[cat_cols])
        # Transform the DataFrame using Label
        train=encoder.transform(X=train)
        
        # Fit the simple imputation method in input columns
        simple_imputer = AutoSimpleImputer(strategy='mean')
        simple_imputer.fit(train)  # Fit on the Train DataFrame
        train = simple_imputer.transform(train.copy())  # Transform the Train DataFrame
        
        # Fit the imputation model
        model = imput_models(train=train,
                             target=target,
                             parameters=imputer_configs,
                             algo=imput_model)
        
        # Store the fitted model information in a dictionary
        if c==0:
            imp_config = {target:{'model_name':imput_model,
                                  'model':model,
                                  'pre_process':encoder,
                                  'imputer':simple_imputer}}
        elif c>0:
            imp_config_2 = {target:{'model_name':imput_model,
                                    'model':model,
                                    'pre_process':encoder,
                                    'imputer':simple_imputer}}
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
        
        encoder=fit_configs[target]['pre_process']
        # Transform the DataFrame using Label
        test_df=encoder.transform(X=test_df)
        
        # Impute the DataFrame using Simple Imputer
        simple_imputer=fit_configs[target]['imputer']
        test_df = simple_imputer.transform(test_df.copy())  
        
        sel_cols=list(test_df.columns)
        sel_cols.remove(target)
        sel_cols.append(target)
        test_df=test_df[sel_cols]
        X_test = test_df.iloc[:, 0:(len(sel_cols)-1)].values

        model=fit_configs[target]['model']
    
        y_predict = model.predict(X_test)

        df_[target].iloc[test_index]=y_predict

    return df_
