def imputer_parameters():
        
    parameters = {
                'RandomForest': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'criterion': "squared_error",
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt'
                },
                'ExtraTrees': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'criterion': "squared_error",
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt'
                },
                'GBR': {
                    'n_estimators': 100,
                    'criterion': "friedman_mse",
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'loss': 'squared_error'
                },
                'KNN': {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'p': 2
                },
                'XGBoost': {
                    'objective': 'reg:squarederror',
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'reg_lambda': 1,
                    'reg_alpha': 0,
                    'subsample': 1,
                    'colsample_bytree': 1
                },
                'Lightgbm': {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'mse',
                    'num_leaves': 31,
                    'max_depth': -1,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'verbose': 0,
                    'force_col_wise': True,
                    'min_data_in_leaf': 20
                },
                'Catboost': {
                    'iterations': 100,
                    'loss_function': 'RMSE',
                    'depth': 8,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 3,
                    'border_count': 254,
                    'subsample': 1
                }
            }
    
    return parameters


