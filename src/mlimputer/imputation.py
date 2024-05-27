import pandas as pd
from tqdm import tqdm
from atlantic.processing.encoders import AutoLabelEncoder
from atlantic.imputers.imputation import AutoSimpleImputer
from mlimputer.parameters import imputer_parameters
from mlimputer.model_selection import missing_report, imput_models

parameters=imputer_parameters()

class MLimputer:
    def __init__ (self, 
                  imput_model : str,
                  imputer_configs : dict=parameters):
        self.imput_model = imput_model
        self.imputer_configs = imputer_configs
        self.imp_config = {}
        self.encoder = None
    
    def fit_imput(self, 
                  X:pd.DataFrame):
        """
        
        This method fits missing data in a dataframe using the imputation method specified by the user.
        
        Parameters:
        X (pd.DataFrame): The input pandas dataframe that needs to be imputed.
        
        """
        
        X_ = X.copy()
        
        X_md ,c = missing_report(X_) ,0
        imp_targets = list(X_md['columns']) 
        
        for col in X_.columns:
            if X_[col].isnull().all():
                raise ValueError(f'Column {col} is filled with null values')
        
        # Iterate over each column with missing data and fit the imputation method
        for target in tqdm(imp_targets, 
                           desc = "Fitting Missing Data", 
                           ncols = 80):
            
            # Split the data into train and test sets
            total_index = X_.index.tolist()
            test_index = X_[X_[target].isnull()].index.tolist()
            train_index = [ value for value in total_index if value not in test_index ]
            
            train = X_.iloc[train_index]
            
            cat_cols = [ col for col in train.select_dtypes(include = ['object','category']).columns if col != target ]
            
            if len(cat_cols) > 0:
                ## Create Label Encoder
                self.encoder = AutoLabelEncoder()
                ## Fit Label Encoder
                self.encoder.fit(X = train[cat_cols])
                # Transform the DataFrame using Label
                train = self.encoder.transform(X = train)
            
            # Fit the simple imputation method in input columns
            simple_imputer = AutoSimpleImputer(strategy = 'mean')
            simple_imputer.fit(train)  # Fit on the Train DataFrame
            train = simple_imputer.transform(train.copy())  # Transform the Train DataFrame
            
            # Fit the imputation model
            model = imput_models(train = train,
                                 target = target,
                                 parameters = self.imputer_configs,
                                 algo = self.imput_model)
            
            # Store the fitted model information in a dictionary
            if c == 0:
                self.imp_config = {target:{'model_name' : self.imput_model,
                                           'model' : model,
                                           'pre_process' : self.encoder,
                                           'imputer' : simple_imputer}}
            elif c > 0:
                imp_config_2 = {target:{'model_name' : self.imput_model,
                                        'model' : model,
                                        'pre_process' : self.encoder,
                                        'imputer' : simple_imputer}}
                self.imp_config.update(imp_config_2)
            c+=1
            
        return self
    
    def transform_imput(self,
                        X : pd.DataFrame):
        """
        Imputation of missing values in a X using a pre-fit imputation model.
        
        Parameters:
        -----------
        X: pd.DataFrame
            The X containing missing values to be imputed.
            
        Returns:
        --------
        X_: pd.DataFrame
            The original X with missing values imputed.
        """
        X_ ,imp_cols = X.copy() ,list(self.imp_config.keys()) 
        
        for col in tqdm(imp_cols, desc = "Imputing Missing Data", ncols = 80):
            
            target = col
            test_index = X_[X_[target].isnull()].index.tolist()
            test = X_.iloc[test_index]
            
            encoder = self.imp_config[target]['pre_process']
            # Transform the DataFrame using Label
            if encoder is not None: test = encoder.transform(X=test)
            
            # Impute the DataFrame using Simple Imputer
            simple_imputer=self.imp_config[target]['imputer']
            test = simple_imputer.transform(test.copy())  
            
            sel_cols = [col for col in test.columns if col != target] + [target]
            test = test[sel_cols]
            X_test = test.iloc[:, 0:(len(sel_cols)-1)].values
    
            model = self.imp_config[target]['model']
        
            y_predict = model.predict(X_test)
    
            X_[target].iloc[test_index] = y_predict
    
        return X_
