from atlantic.analysis import Analysis
from atlantic.evaluation import Evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import catboost as cb
import xgboost
import lightgbm as lgb 
from mlimputer.parameters import imputer_parameters

parameters=imputer_parameters()

def imput_models(train : pd.DataFrame,
                 target : str = "y",
                 algo : str = 'RandomForest',
                 parameters : dict = parameters):
    """
    This function trains and returns a regression model based on the input data and specified algorithm.
    
    Parameters:
    train (pd.DataFrame): The input training data
    target (str, optional): The target column name in the training data. Default is 'y'
    algo (str, optional): The algorithm to be used for training. Default is 'RandomForest'
    parameters (dict, optional): The hyperparameters for the specified algorithm. Default is 'parameters'
    
    Returns:
    model: trained machine learning model.
    """
    
    sel_cols=list(train.columns)
    sel_cols.remove(target)
    sel_cols.append(target)
    train = train[sel_cols]
    
    X_train = train.iloc[:, 0:(len(sel_cols)-1)].values
    y_train = train.iloc[:, (len(sel_cols)-1)].values
    
    if algo == 'RandomForest':
        rf_params = parameters['RandomForest']
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        
    elif algo == 'ExtraTrees':
        et_params = parameters['ExtraTrees']
        model = ExtraTreesRegressor(**et_params)
        model.fit(X_train, y_train)
        
    elif algo == 'GBR':
        gbr_params = parameters['GBR']
        model = GradientBoostingRegressor(**gbr_params)
        model.fit(X_train, y_train)
        
    elif algo == 'KNN':
        knn_params = parameters['KNN']
        model = KNeighborsRegressor(**knn_params)
        model.fit(X_train, y_train)
        
    elif algo == 'XGBoost':
        xg_params = parameters['XGBoost']
        model = xgboost.XGBRegressor(**xg_params)
        model.fit(X_train, y_train)
    
    elif algo == "Catboost":
        cb_params = parameters['Catboost']
        model = cb.CatBoostRegressor(**cb_params)
        model.fit(X_train, y_train)
        
    elif algo == "Lightgbm":
        lb_params = parameters['Lightgbm']
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lb_params, train_data, num_boost_round=100)
        
    else:
        raise ValueError('Invalid model')
   
    return model

def missing_report(X : pd.DataFrame):
    """
    This function generates a report of missing values in a given X.
    
    Parameters:
    X (pd.DataFrame): The input data to be analyzed for missing values
    
    Returns:
    X_md (pd.DataFrame): A dataframe containing the count and percentage of missing values 
    for each numerical column in the input X. 
    Sorted by the percentage of missing values in ascending order. <-***********
    """
    
    X_ = X.copy()
    
    num_cols = X_.select_dtypes(include=['int','float']).columns.tolist()
    X_md = pd.DataFrame(X_[num_cols].isna().sum().loc[X_[num_cols].isna().sum() > 0], columns = ['null_count'])
    X_md['null_percentage'] = X_md['null_count'] / len(X_)
    X_md = X_md.sort_values(by = 'null_percentage', ascending=True)
    X_md['columns'] = X_md.index
    X_md = X_md.reset_index(drop = True)
    
    return X_md

def cross_validation(X : pd.DataFrame, 
                     target : str, 
                     test_size : float = 0.2, 
                     n_splits : int = 5,
                     models : list = []):
    """
    This function performs cross-validation on the given X. The X is divided into training and test sets, 
    and then each model from the list is fit and evaluated on the test set. The performance of each model on the test
    set is recorded in the form of various metrics, such as accuracy, precision, recall, F1-score, etc. (depending on
    whether the target variable is a continuous variable or a categorical variable). The final result of the function
    is a leaderboard containing the metrics of each model for each fold of the cross-validation.
    
    Parameters:
    -----------
    X: pd.DataFrame
        The input X to be evaluated.
    target: str
        The name of the target column.
    test_size: float, optional (default=0.2)
        The size of the test set, specified as a fraction of the input X.
    n_splits: int, optional (default=5)
        The number of folds for cross-validation.
    models: list, optional (default=[])
        A list of models to be evaluated.
        
    Returns:
    --------
    leaderboard: pd.DataFrame
        A dataframe containing the performance metrics of each model for each fold of the cross-validation.
    """
    
    y ,list_metrics = X[target], []
    sv_pred ,_ = Analysis(target=target).target_type(X = X)
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        print(f"Fold {i + 1} :")

        X_train,X_test = X_train.drop([target], axis = 1),X_test.drop([target], axis=1)
        for model in models:
            print(f"Fitting {model.__class__.__name__} model")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = model.score(X_test, y_test)
            print(f"{model.__class__.__name__} model score: {score}")
            if sv_pred == 'Class':
                metrics = pd.DataFrame(Evaluation.metrics_classification(y_test,y_pred),index = [0])
            elif sv_pred == 'Reg':
                metrics = pd.DataFrame(Evaluation.metrics_regression(y_test,y_pred),index = [0])
            metrics["model"] = model.__class__.__name__
            metrics["n_splits"] = i+1
            metrics = metrics.reset_index(drop = True)
            list_metrics.append(metrics)

    leaderboard = pd.concat(list_metrics)
    
    leaderboard = leaderboard.reset_index(drop = True)
    
    return leaderboard