<br>
<p align="center">
  <h2 align="center"> MLimputer - Null Imputation Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `MLimputer` project constitutes an complete and integrated pipeline to automate the handling of missing values in Datasets through regression prediction and aims at reducing bias and increase the precision of imputation results when compared to more classic single imputation methods.
This package provides multiple algorithm options to impute your data (shown bellow), in which every observed data column with existing missing values is fitted and subsequently predicted with a robust preprocessing approach.

The architecture design includes three main sections, these being: missing data analysis, data preprocessing and predictive method selection which are organized in a pipeline structure.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed imputation procedures are applicable on any data table associated with any Supervised ML scopes, based on missing data columns to be imputed.
    
* Robustness and improvement of predictive results: The application of the MLimputer preprocessing aims at improve the predictive performance through optimized imputation of existing missing values in the Dataset input columns. 
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 

* [Pandas](https://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [CatBoost](https://catboost.ai/)
    
## Where to get it <a name = "ta"></a>
    
Binary installer for the latest released version is available at the Python Package Index (PyPI).   

## Installation  

To install this package from Pypi repository run the following command:

```
pip install mlimputer
```

# Usage Examples
    
The first needed step after importing the package is to load a dataset (split it) and define your choosen imputation model in`fit_imput` function.
The imputation model options for handling the missing data in your dataset are the following:
* `RandomForest`
* `ExtraTrees`
* `GBR`
* `KNN`
* `GeneralizedLR`
* `XGBoost`
* `Lightgbm`
* `Catboost`

After fitting your imputation model, you can load the `imputer` variable into `fit_configs` parameter in the `transform_imput` function. From there you can impute the future datasets (validate, test ...) with the same data properties. 

Through the `cross_validation` function you can also compare the predictive performance evalution of multiple imputations, allowing you to validate which imputation model fits better your future predictions.

Importante Notes:

* The actual version of this package does not incorporate the imputing of categorical values, just the automatic handling of numeric missing values is implemented.

    
```py

    
import mlimputer as mli
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

data = pd.read_csv('csv_directory_path') # Dataframe Loading Example

train, test= train_test_split(data, train_size=0.8)
train,test=train.reset_index(drop=True), test.reset_index(drop=True) # <- importante
    
imp_model="RandomForest"  
# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","GeneralizedLR","XGBoost","Lightgbm","Catboost"

# Imputation Example 1 : RandomForest

imputer_rf=mli.fit_imput(Dataset=train,imput_model="RandomForest")
train_rf=mli.transform_imput(Dataset=train,fit_configs=imputer_rf)
test_rf=mli.transform_imput(Dataset=test,fit_configs=imputer_rf)

# Imputation Example 2 : XGBoost

imputer_xgb=mli.fit_imput(Dataset=train,imput_model="XGBoost")
train_xgb=mli.transform_imput(Dataset=train,fit_configs=imputer_xgb)
test_xgb=mli.transform_imput(Dataset=test,fit_configs=imputer_xgb)

#(...)
    
## Performance Evaluation Example - Imputation CrossValidation

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
        
leaderboard_xgb_imp=mli.cross_validation(Dataset=train_xgb,
                                         target="Target_Name_Col", 
                                         test_size=0.2,
                                         n_splits=3,
                                         models=[LinearRegression(), RandomForestRegressor(), CatBoostRegressor()])

## Export Imputation Metadata

# XGBoost Imputation Metadata
import pickle 
output = open("imputer_xgb.pkl", 'wb')
pickle.dump(imputer_xgb, output)
    

```  
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/TSForecasting/blob/main/LICENSE) for more information.

## Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
    
Feel free to contact me and share your feedback.
