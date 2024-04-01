def imputer_parameters():
    
    parameters ={'RandomForest':{'n_estimators':50,'random_state':42,'criterion':"squared_error",
                                 'max_depth':None},
                 'ExtraTrees':{'n_estimators':50,'random_state':42,'criterion':"squared_error",
                               'max_depth':None}, 
                 'GBR':{'n_estimators':50,'learning_rate':0.01,'criterion':"friedman_mse",
                        'max_depth':3,'min_samples_split':5,'loss':'squared_error'},
                 'KNN':{'n_neighbors': 3,'weights':"uniform",'algorithm':"auto",
                        'metric_params':None},
                 'XGBoost':{'objective':'reg:squarederror','n_estimators':100,'nthread':24},
                 "Lightgbm": {'boosting_type': 'gbdt','objective':'regression','metric': 'mse',
                              'num_leaves': 31,'learning_rate': 0.05,'feature_fraction': 0.9,
                              'bagging_fraction': 0.8,'bagging_freq': 5}, 
                 'Catboost':{'iterations': 100, 'depth': 5, 'learning_rate': 0.1, 'loss_function': 'RMSE', 
                             'l2_leaf_reg': 3, 'random_seed': 42, 'verbose': False }
                 }
    return parameters