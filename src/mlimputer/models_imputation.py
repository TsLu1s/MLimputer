import lightgbm as lgb
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor

class RandomForestImputation:
    """
    This class encapsulates a Random Forest regression model tailored for imputation tasks,
    with hyperparameters specifically configured for effective operation.
    """
    def __init__(self, n_estimators=100, random_state = 42, criterion = "squared_error", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        """
        Initialize the RandomForestImputation model with the specified hyperparameters.

        Parameters:
            n_estimators (int): Number of trees in the forest.
            max_depth (int or None): Maximum depth of each tree. None means nodes expand until all leaves are pure.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
            max_features (str or int): Number of features to consider when looking for the best split; "auto" uses all features.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, criterion=self.criterion,
                                           max_depth=self.max_depth, min_samples_split=self.min_samples_split, 
                                           min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)

    def fit(self, X, y):
        """
        Fit the RandomForest model to the provided training data.

        Parameters:
            X (array-like): Feature dataset for training.
            y (array-like): Target values.
        """
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        """
        Predict using the fitted RandomForest model.

        Parameters:
            X (array-like): Dataset for which to make predictions.

        Returns:
            array: Predicted values.
        """
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            dict: Parameters of this estimator.
        """
        return {"n_estimators": self.n_estimators, "random_state": self.random_state,"criterion": self.criterion,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split, 
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features}

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Parameters:
            parameters (dict): Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        """
        Verify that the model has been fitted.

        Raises:
            AssertionError: If the model is not fitted.
        """
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."
        
class ExtraTreesImputation:
    """
    This class encapsulates an Extra Trees regression model optimized for imputation tasks,
    with hyperparameters specifically configured to effectively handle complex data structures.
    """
    def __init__(self, n_estimators=100, random_state = 42, criterion = "squared_error", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        """
        Initialize the ExtraTreesImputation model with the specified hyperparameters.

        Parameters:
            n_estimators (int): The number of trees in the forest.
            max_depth (int or None): The maximum depth of the tree. If None, then nodes are expanded until all leaves contain less than min_samples_split samples.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (str or int): The number of features to consider when looking for the best split.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = ExtraTreesRegressor(n_estimators=self.n_estimators, random_state=self.random_state, criterion=self.criterion,
                                         max_depth=self.max_depth, min_samples_split=self.min_samples_split, 
                                         min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "random_state": self.random_state, "criterion": self.criterion,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split, 
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class GBRImputation:
    """
    This class encapsulates a Gradient Boosting regression model designed for imputation, 
    integrating advanced configurations to handle various data anomalies and patterns.
    """
    def __init__(self, n_estimators=100, criterion = "friedman_mse", learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, loss = 'squared_error'):
        """
        Initialize the GBRImputation model with the specified hyperparameters.

        Parameters:
            n_estimators (int): The number of boosting stages to be run.
            learning_rate (float): Rate at which the contribution of each tree is shrunk.
            max_depth (int): Maximum depth of the individual regression estimators.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, criterion=self.criterion, learning_rate=self.learning_rate,
                                               max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                               min_samples_leaf=self.min_samples_leaf, loss=self.loss)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "criterion": self.criterion, "learning_rate": self.learning_rate,
                "max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf , 'loss': self.loss}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class KNNImputation:
    """
    This class wraps a K-Nearest Neighbors regressor for use in imputation tasks, providing a simple yet effective approach to handling missing data.
    """
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2):
        """
        Initialize the KNNImputation model with the specified hyperparameters.

        Parameters:
            n_neighbors (int): Number of neighbors to use for kneighbors queries.
            weights (str): Weight function used in prediction. Possible values: 'uniform', 'distance'.
            algorithm (str): Algorithm used to compute the nearest neighbors. Can be 'ball_tree', 'kd_tree', or 'auto'.
            leaf_size (int): Leaf size passed to BallTree or KDTree.
            p (int): Power parameter for the Minkowski metric.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, 
                                         algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"n_neighbors": self.n_neighbors, "weights": self.weights, 
                "algorithm": self.algorithm, "leaf_size": self.leaf_size, "p": self.p}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class XGBoostImputation:
    """
    This class encapsulates an XGBoost regression model tailored for imputation,
    leveraging powerful gradient boosting techniques to handle various types of data with high efficiency.
    """
    def __init__(self, n_estimators=100, objective = 'reg:squarederror', learning_rate=0.1, max_depth=3, 
                 reg_lambda=1, reg_alpha=0, subsample=1, colsample_bytree=1):
        """
        Initialize the XGBoostImputation model with the specified hyperparameters.

        Parameters:
            n_estimators (int): Number of gradient boosted trees. Equivalent to the number of boosting rounds.
            learning_rate (float): Step size shrinkage used to prevent overfitting. Range is [0,1].
            max_depth (int): Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
            reg_lambda (float): L2 regularization term on weights. Increasing this value will make model more conservative.
            reg_alpha (float): L1 regularization term on weights. Increasing this value will make model more conservative.
            subsample (float): Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
            colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        """
        self.n_estimators = n_estimators
        self.objective = objective
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = xgb.XGBRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                      max_depth=self.max_depth, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha,
                                      subsample=self.subsample, colsample_bytree=self.colsample_bytree, verbosity=0)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators, "objective": self.objective, "learning_rate": self.learning_rate, "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda, "reg_alpha": self.reg_alpha, "subsample": self.subsample, 
            "colsample_bytree": self.colsample_bytree
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class CatBoostImputation:
    """
   This class encapsulates a CatBoost regression model, ideal for imputation with its robust handling of categorical data,
   and efficient processing capabilities that minimize overfitting while maximizing predictive performance.
   """
    def __init__(self, iterations=100, loss_function = 'RMSE', depth=8, learning_rate=0.1, l2_leaf_reg=3, 
                 border_count=254, subsample=1):
        """
        Initialize the CatBoostImputation model with the specified hyperparameters.

        Parameters:
            iterations (int): The maximum number of trees that can be built when solving machine learning problems.
            depth (int): Depth of the tree. A deep tree can model more complex relationships by adding more splits; it also risks overfitting.
            learning_rate (float): The learning rate used in updating the model as it attempts to minimize the loss function.
            l2_leaf_reg (float): Coefficient at the L2 regularization term of the cost function, which controls the trade-off between achieving lower training error and reducing model complexity to avoid overfitting.
            border_count (int): The number of splits for numerical features used to find the optimal cut points.
            subsample (float): The subsample ratio of the training instance. Setting it lower can prevent overfitting but may raise the variance of the model.
        """
        self.iterations = iterations
        self.loss_function = loss_function
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.subsample = subsample
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            loss_function = self.loss_function,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            subsample=self.subsample,
            save_snapshot=False,
            verbose=False
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X):
        self.check_is_fitted()
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "iterations": self.iterations, "depth": self.depth, "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg, "border_count": self.border_count, "subsample": self.subsample
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.is_fitted_ = False
        return self

    def check_is_fitted(self):
        assert hasattr(self, 'is_fitted_'), "Model is not fitted yet."

class LightGBMImputation:
    """
    This class encapsulates a LightGBM regression model, well-suited for imputation tasks with large datasets,
    utilizing light gradient boosting mechanism that is resource-efficient yet delivers high performance.
    """

    def __init__(self, boosting_type='gbdt', objective='regression', metric = 'mse', num_leaves=31, max_depth=-1, learning_rate=0.1, 
                 n_estimators=100, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, 
                 reg_lambda=0.1, verbose=0, force_col_wise=True, min_data_in_leaf=20):
        """
        Initialize the LightGBMImputation model with the specified hyperparameters.

        Parameters:
            boosting_type (str): Type of boosting to perform.
            objective (str): The optimization objective of the model.
            num_leaves (int): Maximum number of leaves in one tree.
            max_depth (int): Maximum tree depth for base learners.
            learning_rate (float): Boosting learning rate.
            n_estimators (int): Number of boosted trees to fit.
            feature_fraction (float): Fraction of features to be used in each iteration.
            bagging_fraction (float): Fraction of data to be used for each tree.
            bagging_freq (int): Frequency of bagging.
            reg_alpha (float): L1 regularization term.
            reg_lambda (float): L2 regularization term.
            verbose (int): Verbosity for logging.
            force_col_wise (bool): Force col-wise histogram building.
            min_data_in_leaf (int): Minimum number of samples in one leaf.
        """
        self.boosting_type = boosting_type
        self.objective = objective
        self.metric = metric
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.force_col_wise = force_col_wise
        self.min_data_in_leaf = min_data_in_leaf
        self.params = {
            "boosting_type": self.boosting_type, "objective": self.objective, 'metric': self.metric, "num_leaves": self.num_leaves,
            "max_depth": self.max_depth, "learning_rate": self.learning_rate, "n_estimators": self.n_estimators,
            "feature_fraction": self.feature_fraction, "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq, "reg_alpha": self.reg_alpha, "reg_lambda": self.reg_lambda,
            "verbose": self.verbose, "force_col_wise": self.force_col_wise, "min_data_in_leaf": self.min_data_in_leaf}

    def fit(self, X, y):
        """
        Fit the LightGBM model to the provided training data using the specified parameters.

        Parameters:
            X (array-like): Feature dataset for training.
            y (array-like): Target values.
        """
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, train_data, num_boost_round=self.n_estimators)

    def predict(self, X):
        """
        Predict using the fitted LightGBM model.

        Parameters:
            X (array-like): Dataset for which to make predictions.

        Returns:
            array: Predicted values.
        """
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return self.params

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for key, value in parameters.items():
            if key in self.params:
                setattr(self, key, value)
                self.params[key] = value
        self.is_fitted_ = False  # Invalidate the model fitting
        return self

    def check_is_fitted(self):
        """
        Verify that the model has been fitted.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise AttributeError("This LightGBMImputation instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
    




















