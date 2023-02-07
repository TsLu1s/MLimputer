import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
import atlantic as atl
import catboost as cb
import xgboost
import lightgbm as lgb 

parameters ={'RandomForest':{'n_estimators':250,'random_state':42,'criterion':"squared_error",
                             'max_depth':None,'max_features':"auto"},
             'ExtraTrees':{'n_estimators':250,'random_state':42,'criterion':"squared_error",
                           'max_depth':None,'max_features':"auto"}, 
             'GBR':{'n_estimators':250,'learning_rate':0.1,'criterion':"friedman_mse",
                    'max_depth':3,'min_samples_split':5,'learning_rate':0.01,'loss':'ls'},
             'KNN':{'n_neighbors': 3,'weights':"uniform",
                    'algorithm':"auto",'metric_params':None},
             'GeneralizedLR':{'power':1,'alpha':0.5,'link':'log','fit_intercept':True,
                              'max_iter':100,'warm_start':False,'verbose':0},
             'XGBoost':{'objective':'reg:squarederror','n_estimators':1000,'nthread':24},
             "Lightgbm": {'boosting_type': 'gbdt','objective':'regression','metric': 'mse',
                          'num_leaves': 31,'learning_rate': 0.05,'feature_fraction': 0.9,
                          'bagging_fraction': 0.8,'bagging_freq': 5,'reg_alpha': 0.1,
                          'reg_lambda': 0.1,'verbose': 0}, 
             'AutoKeras':{'max_trials':1,'overwrite':42,'loss':"mean_squared_error",
                          'max_model_size':None,'epochs':50},
             'Catboost':{'verbose':False}
             }

def imput_models(Train:pd.DataFrame,
                 target:str="y",
                 algo:str='RandomForest',
                 parameters:dict=parameters):
    """
    This function trains and returns a regression model based on the input data and specified algorithm.
    
    Parameters:
    Train (pd.DataFrame): The input training data
    target (str, optional): The target column name in the training data. Default is 'y'
    algo (str, optional): The algorithm to be used for training. Default is 'RandomForest'
    parameters (dict, optional): The hyperparameters for the specified algorithm. Default is 'parameters'
    
    Returns:
    model: Trained machine learning model.
    """
    
    sel_cols=list(Train.columns)
    sel_cols.remove(target)
    sel_cols.append(target)
    Train=Train[sel_cols]
    
    X_train = Train.iloc[:, 0:(len(sel_cols)-1)].values
    y_train = Train.iloc[:, (len(sel_cols)-1)].values
    
    if algo=='RandomForest':
        rf_params=parameters['RandomForest']
        model = RandomForestRegressor(**rf_params) 
        model.fit(X_train, y_train)
        
    elif algo=='ExtraTrees':
        et_params=parameters['ExtraTrees']
        model = ExtraTreesRegressor(**et_params)
        model.fit(X_train, y_train)
        
    elif algo=='GBR':
        gbr_params=parameters['GBR']
        model = GradientBoostingRegressor(**gbr_params)
        model.fit(X_train, y_train)
        
    elif algo=='KNN':
        knn_params=parameters['KNN']
        model = KNeighborsRegressor(**knn_params)
        model.fit(X_train, y_train)
        
    elif algo=='GeneralizedLR':
        td_params=parameters['GeneralizedLR']
        model = TweedieRegressor(**td_params)
        model.fit(X_train, y_train)
        
    elif algo=='XGBoost':
        xg_params=parameters['XGBoost']
        model = xgboost.XGBRegressor(**xg_params)
        model.fit(X_train, y_train)
    
    elif algo=="Catboost":
        cb_params=parameters['Catboost']
        model = cb.CatBoostRegressor(**cb_params)
        model.fit(X_train, y_train)
    elif algo=="Lightgbm":
        lb_params=parameters['Lightgbm']
        model = lgb.train(lb_params, X_train, num_boost_round=100)
        
    else:
        raise ValueError('Invalid model')
   
    return model

def missing_report(Dataset:pd.DataFrame):
    """
    This function generates a report of missing values in a given dataset.
    
    Parameters:
    Dataset (pd.DataFrame): The input data to be analyzed for missing values
    
    Returns:
    df_md (pd.DataFrame): A dataframe containing the count and percentage of missing values 
    for each numerical column in the input dataset. 
    Sorted by the percentage of missing values in ascending order. <-***********
    """
    
    df=Dataset.copy()
    
    num_cols=df.select_dtypes(include=['int','float']).columns.tolist()
    df_md = pd.DataFrame(df[num_cols].isna().sum().loc[df[num_cols].isna().sum() > 0], columns=['missing_data_count'])
    df_md['missing_data_percentage'] = df_md['missing_data_count'] / len(df)
    df_md = df_md.sort_values(by='missing_data_percentage', ascending=True)
    df_md['columns']=df_md.index
    df_md=df_md.reset_index(drop=True)
    
    return df_md

def fit_imput(Dataset:pd.DataFrame,
              imput_model:str="KNN"):
              #,imputation_order="ascending","descending","random"):
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
    imp_targets=list(df_md['columns'])    #,input_strategy="mean","median","most_frequent"):
        
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
                             parameters=parameters,
                             algo='KNN')
        
        # Store the fitted model information in a dictionary
        if c==0:
            imp_config = {Target:{'model':model,
                                  'pre_process':le_fit,
                                  'input_nulls':imputer_simple}}
        elif c>0:
            imp_config_2 = {Target:{'model':model,
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

def cross_validation(Dataset:pd.DataFrame, target:str, test_size:float=0.2, n_splits:int=5,models:list=[]):#=[LinearRegression(), RandomForestRegressor(), CatBoostRegressor()]):
    """
    This function performs cross-validation on the given dataset. The dataset is divided into training and test sets, 
    and then each model from the list is fit and evaluated on the test set. The performance of each model on the test
    set is recorded in the form of various metrics, such as accuracy, precision, recall, F1-score, etc. (depending on
    whether the target variable is a continuous variable or a categorical variable). The final result of the function
    is a leaderboard containing the metrics of each model for each fold of the cross-validation.
    
    Parameters:
    -----------
    Dataset: pd.DataFrame
        The input dataset to be evaluated.
    target: str
        The name of the target column.
    test_size: float, optional (default=0.2)
        The size of the test set, specified as a fraction of the input dataset.
    n_splits: int, optional (default=5)
        The number of folds for cross-validation.
    models: list, optional (default=[])
        A list of models to be evaluated.
        
    Returns:
    --------
    leaderboard: pd.DataFrame
        A dataframe containing the performance metrics of each model for each fold of the cross-validation.
    """
    
    y,list_metrics=Dataset[target],[]
    sv_pred=atl.target_type(Dataset, target)[0]
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(Dataset, y, test_size=test_size)
        print(f"Fold number: {i + 1}")

        X_train,X_test=X_train.drop([target], axis=1),X_test.drop([target], axis=1)
        for model in models:
            print(f"Fitting {model.__class__.__name__} model")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = model.score(X_test, y_test)
            print(f"{model.__class__.__name__} model score: {score}")
            if sv_pred=='Class':
                metrics=pd.DataFrame(atl.metrics_classification(y_test,y_pred),index=[0])
            elif sv_pred=='Reg':
                metrics = pd.DataFrame(atl.metrics_regression(y_test,y_pred),index=[0])
            metrics["model"]=model.__class__.__name__
            metrics["n_splits"]=i+1
            metrics=metrics.reset_index(drop=True)
            list_metrics.append(metrics)

    leaderboard= pd.concat(list_metrics)
    leaderboard=leaderboard.reset_index(drop=True)
    return leaderboard
