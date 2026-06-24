from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from data.feature_engineering import Feature_preprocess

import numpy as np

class PredictiveModel:
    def __init__(self, config, model_params=None):
        self.config = config
        self.data_path = config.data_path
        self.feature_engineer = Feature_preprocess(self.config)
        self.RANDOM_STATE = config.random_state
        
        default_params = {
            'n_estimators': 100, 
            'learning_rate': 0.05,
            'max_depth': 5, 
            'random_state': self.RANDOM_STATE,
            'n_jobs': -1
        }
        if model_params:
            default_params.update(model_params)
            
        self.model = XGBRegressor(**default_params)

    def train_cv(self, window=20, threshold=125, n_splits=5):
        train_df, test_df, act_df = self.feature_engineer.load_data()
        X_train, y_train, _, _ = self.feature_engineer.test_train_split(train_df, test_df, act_df, threshold=threshold, window=window)

        group=train_df['unit_number']
        gkf = GroupKFold(n_splits=n_splits)
        
        print(f"Performing cross-validation (Window Size: {window}, Target Clip: {threshold})...")
        cv_scores = cross_val_score(
            self.model, 
            X_train, 
            y_train, 
            cv=gkf, 
            groups=group,
            scoring='neg_root_mean_squared_error')
        
        return float(np.mean(-cv_scores))

    def train_and_evaluate(self, window=20, threshold=125):
        train_df, test_df, act_df = self.feature_engineer.load_data()
        
        X_train, y_train, X_test, y_test = self.feature_engineer.test_train_split(
            train_df, test_df, act_df, threshold=threshold, window=window
        )
        
        print(f"Training XGBRegressor (Window Size: {window}, Target Clip: {threshold})...")
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, predictions)
        print(f'Pipeline Complete! Test RMSE: {rmse:.2f} cycles')
        
        return y_test, predictions, rmse
    