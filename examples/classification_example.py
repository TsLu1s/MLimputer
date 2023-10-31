import mlimputer as mli
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

# Source Data: https://github.com/airtlab/machine-learning-for-quality-prediction-in-plastic-injection-molding

url="https://github.com/TsLu1s/MLimputer/blob/main/data/data_injection_quality.csv"

data = pd.read_csv('DTx Dev/dev/data/data_mold_optimization.csv', encoding='latin', delimiter=';')
target="quality"

## Generate Random Null Values in Dataset
missing_ratio = 0.1  # 10% missing values
missing_mask = np.random.rand(*data.shape) < missing_ratio
data = data.copy()
data[missing_mask] = np.nan
data.isna().sum()

data=data[data[target].isnull()==False]
data=data.reset_index(drop=True)

train,test = train_test_split(data, train_size=0.7)
train,test = train.reset_index(drop=True), test.reset_index(drop=True)

# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"

# Imputation Example 1 : RandomForest

imputer_knn=mli.fit_imput(dataset=train,imput_model="KNN")
train_knn=mli.transform_imput(dataset=train,fit_configs=imputer_knn)
test_knn=mli.transform_imput(dataset=test,fit_configs=imputer_knn)

# Imputation Example 2 : XGBoost

imputer_xgb=mli.fit_imput(dataset=train,imput_model="XGBoost")
train_xgb=mli.transform_imput(dataset=train,fit_configs=imputer_xgb)
test_xgb=mli.transform_imput(dataset=test,fit_configs=imputer_xgb)

# Imputation Example 3 : GBR

imputer_gbr=mli.fit_imput(dataset=train,imput_model="GBR")
train_gbr=mli.transform_imput(dataset=train,fit_configs=imputer_gbr)
test_gbr=mli.transform_imput(dataset=test,fit_configs=imputer_gbr)

# Imputation Example 4 : Lightgbm

imputer_lgb=mli.fit_imput(dataset=train,imput_model="Lightgbm")
train_lgb=mli.transform_imput(dataset=train,fit_configs=imputer_lgb)
test_lgb=mli.transform_imput(dataset=test,fit_configs=imputer_lgb)

## Performance Evaluation Example - Imputation CrossValidation

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from atlantic.processing import AutoLabelEncoder

## Preprocessing Data (Label Encoder)

cat_cols=[col for col in train_xgb.select_dtypes(include=['object']).columns if col != target]

## Preprocessing Data (Label Encoder)
encoder = AutoLabelEncoder()
encoder.fit(train_xgb[cat_cols])
df=encoder.transform(X=train_xgb)
df=df.reset_index(drop=True)
df[target]=df[target].astype(str)

leaderboard_xgb_imp=mli.cross_validation(dataset=df,
                                         target=target,
                                         test_size=0.2,
                                         n_splits=3,
                                         models=[RandomForestClassifier(), DecisionTreeClassifier()])
