import mlimputer as mli
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

#source_data="https://www.kaggle.com/datasets/surekharamireddy/fraudulent-claim-on-cars-physical-damage"

url="https://raw.githubusercontent.com/TsLu1s/Atlantic/main/data/Fraudulent_Claim_Cars_class.csv"
data= pd.read_csv(url) # Dataframe Loading Example

data.isna().sum()
data.dtypes

target="fraud"
data[target]=data[target].astype('category')
data=data[data[target].isnull()==False]
data=data.reset_index(drop=True)

train,test = train_test_split(data, train_size=0.8) 
train,test = train.reset_index(drop=True), test.reset_index(drop=True)

# All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"

# Imputation Example 1 : RandomForest

parameters_=mli.imputer_parameters()
## Customizing parameters settings

parameters_["RandomForest"]["n_estimators"]=40
parameters_["KNN"]["n_neighbors"]=5
print(parameters_)

imputer_rf=mli.fit_imput(Dataset=train,imput_model="RandomForest",imputer_configs=parameters_)
train_rf=mli.transform_imput(Dataset=train,fit_configs=imputer_rf)
test_rf=mli.transform_imput(Dataset=test,fit_configs=imputer_rf)

# Imputation Example 2 : KNN

imputer_knn=mli.fit_imput(Dataset=train,imput_model="KNN",imputer_configs=parameters_)
train_knn=mli.transform_imput(Dataset=train,fit_configs=imputer_knn)
test_knn=mli.transform_imput(Dataset=test,fit_configs=imputer_knn)
    
## Performance Evaluation Example - Imputation CrossValidation

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import atlantic as atl

## Preprocessing Data (Label Encoder)

le_fit=atl.fit_Label_Encoding(train_knn,target)
df=atl.transform_Label_Encoding(train_knn,le_fit)
df=df.reset_index(drop=True)
df[target]=df[target].astype('category')

leaderboard_knn_imp=mli.cross_validation(Dataset=df,
                                         target=target, 
                                         test_size=0.2,
                                         n_splits=3,
                                         models=[XGBClassifier(), RandomForestClassifier(), DecisionTreeClassifier()])


#import pickle 
#output = open("imputer_knn.pkl", 'wb')
#pickle.dump(imputer_knn, output)

