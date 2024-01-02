from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from atlantic.processing import AutoLabelEncoder
from mlimputer.imputation import MLimputer
import mlimputer.model_selection as ms
import mlimputer.parameters as params
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console

########################################## Dataset 1
sel_dataset="Dataset 1"

if sel_dataset=="Dataset 1":
    # Source Data: https://github.com/airtlab/machine-learning-for-quality-prediction-in-plastic-injection-molding
    url='https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/injection_quality.csv'
    data = pd.read_csv(url, encoding='latin', delimiter=';')
    target="quality"
    data[target]=data[target].astype('object')

########################################## Dataset 2

elif sel_dataset=="Dataset 2":
    # Source Data: "https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset"
    data=pd.read_csv('https://github.com/TsLu1s/MLimputer/edit/main/data/body_measurement.csv', encoding='latin', delimiter=',') 
    target="BodyFat"

########################################## Dataset 3

elif sel_dataset=="Dataset 3":
    # Source Data: "https://www.kaggle.com/code/sagardubey3/admission-prediction-with-linear-regression"
    data=pd.read_csv('https://raw.githubusercontent.com/TsLu1s/MLimputer/main/data/Admission_Predict.csv', encoding='latin', delimiter=',') 
    target="Chance of Admit "

## Generate Random Null Values in Dataset
sel_cols = [col for col in data.columns if col != target] + [target]
data = data[sel_cols]
missing_ratio = 0.1 # 10% missing values
for col in sel_cols[:-1]:
        missing_mask = np.random.rand(data.shape[0]) < missing_ratio
        data.loc[missing_mask, col] = np.nan

##############################################################################################################################
##### Preprocessing Data
data = data[data[target].isnull() == False]
data = data.reset_index(drop = True)

train,test = train_test_split(data, train_size = 0.8)
train,test = train.reset_index(drop = True), test.reset_index(drop = True) # -> Required 
train.isna().sum(), test.isna().sum()

## Customizing parameters example
hparameters=params.imputer_parameters()
print(hparameters)
hparameters["RandomForest"]["n_estimators"] = 15
hparameters["ExtraTrees"]["n_estimators"] = 15
hparameters["GBR"]["n_estimators"] = 15
hparameters["KNN"]["n_neighbors"] = 5
hparameters["Lightgbm"]["learning_rate"] = 0.01
hparameters["Catboost"]["loss_function"] = "MAE"

### Performance Evaluation Example - Imputation CrossValidation [Classification, Regression]
# All model imputation options

list_imputers,list_leaderboards= ["RandomForest" , "ExtraTrees", "GBR", "KNN", 
                                  "XGBoost", "Lightgbm", "Catboost"] ,[]

for imput_model in list_imputers:
    
    mli = MLimputer(imput_model = imput_model, imputer_configs = hparameters)
    mli.fit_imput(X=train)
    train_imp = mli.transform_imput(X = train)
    test_imp = mli.transform_imput(X = test)
    
    ## Preprocessing Data (Label Encoder)
    cat_cols = [col for col in data.select_dtypes(include=['object','category']).columns if col != target]
    X = train_imp.copy()
    X.isna().sum()
    
    if len(cat_cols) > 0 :
        ## Create Label Encoder
        encoder = AutoLabelEncoder()
        ## Fit
        encoder.fit(X[cat_cols])
        # Transform the DataFrame using Label Encoding
        X = encoder.transform(X = X)
        
    X = X.reset_index(drop = True)

    if X[target].dtypes == "object":
        
        X[target] = X[target].astype('category')
        leaderboard_imp = ms.cross_validation(X = X,
                                                 target = target,
                                                 test_size = 0.25,
                                                 n_splits = 3,
                                                 models = [RandomForestClassifier(),
                                                           DecisionTreeClassifier()])
        
    elif X[target].dtypes == "int64" or X[target].dtypes == "float64":
        
        leaderboard_imp = ms.cross_validation(X = X,
                                              target = target,
                                              test_size = 0.25,
                                              n_splits = 3,
                                              models = [XGBRegressor(),
                                                        RandomForestRegressor()])
    
    leaderboard_imp['imputer_model'] = imput_model
    list_leaderboards.append(leaderboard_imp)

leaderboard = pd.concat(list_leaderboards)

#import pickle 
#output = open("imputer.pkl", 'wb')
#pickle.dump(imputer_imp, output)
