<br>
<p align="center">
  <h2 align="center"> MLimputer - Null Imputation Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `MLimputer` project constitutes an complete and integrated pipeline to automate the handling of missing values in datasets through regression prediction and aims at reducing bias and increase the precision of imputation results when compared to more classic imputation methods.
This package provides multiple algorithm options to impute your data, in which every observed data column with existing missing values is fitted with a robust preprocessing approach and subsequently predicted.

The architecture design includes three main sections, these being: missing data analysis, data preprocessing and supervised model imputation which are organized in a customizable pipeline structure.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed imputation procedures are applicable on any data table associated with any Supervised ML scopes, based on missing data columns to be imputed.
    
* Robustness and improvement of predictive results: The application of the MLimputer preprocessing aims at improve the predictive performance through customization and optimization of existing missing values imputation in the dataset input columns. 
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 

* [Pandas](https://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [CatBoost](https://catboost.ai/)
    
## Where to get it <a name = "ta"></a>
    
Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/mlimputer/).   

## Installation  

To install this package from Pypi repository run the following command:

```
pip install mlimputer
```

# Usage Examples
    
The first needed step after importing the package is to load a dataset (split it) and define your choosen imputation model.
The imputation model options for handling the missing data in your dataset are the following:
* `RandomForest`
* `ExtraTrees`
* `GBR`
* `KNN`
* `XGBoost`
* `Lightgbm`
* `Catboost`

After creating a `MLimputer` object with your imputation selected model, you can then fit the missing data through the `fit_imput` method. From there you can impute the future datasets with `transform_imput` (validate, test ...) with the same data properties. Note, as it shows in the example bellow, you can also customize your model imputer parameters by changing it's configurations and then, implementing them in the `imputer_configs` parameter.

Through the `cross_validation` function you can also compare the predictive performance evalution of multiple imputations, allowing you to validate which imputation model fits better your future predictions.

Importante Notes:

* The actual version of this package does not incorporate the imputing of categorical values, just the automatic handling of numeric missing values is implemented.

```py

from mlimputer.imputation import MLimputer
import mlimputer.model_selection as ms
import mlimputer.parameters as params
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

data = pd.read_csv('csv_directory_path') # Dataframe Loading Example

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True) # <- Required

# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"

# Customizing Hyperparameters Example

hparameters = params.imputer_parameters()
print(hparameters)
hparameters["KNN"]["n_neighbors"] = 5
hparameters["RandomForest"]["n_estimators"] = 30
    
# Imputation Example 1 : KNN

mli = MLimputer(imput_model = "KNN", imputer_configs = hparameters)
mli.fit_imput(X = train)
train_knn = mli.transform_imput(X = train)
test_knn = mli.transform_imput(X = test)

# Imputation Example 2 : RandomForest

mli = MLimputer(imput_model = "RandomForest", imputer_configs = hparameters)
mli.fit_imput(X = train)
train_rf = mli.transform_imput(X = train)
test_rf = mli.transform_imput(X = test)
    
#(...)
    
## Performance Evaluation Regression - Imputation CrossValidation Example

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
        
leaderboard_knn_imp=ms.cross_validation(X = train_knn,
                                        target = "Target_Name_Col", 
                                        test_size = 0.2,
                                        n_splits = 3,
                                        models = [LinearRegression(), RandomForestRegressor(), CatBoostRegressor()])

## Export Imputation Metadata

# KNN Imputation Metadata
import pickle 
output = open("imputer_knn.pkl", 'wb')
pickle.dump(imputer_knn, output)

```  
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/TSForecasting/blob/main/LICENSE) for more information.

## Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)
