import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from data.feature_engineering import FeaturePreprocess

class PredictiveModel:
    def __init__(self, data_path, model_params=None):
        self.data_path = data_path
        self.feature_engineer = FeaturePreprocess(data_path)
        
        default_params = {
            'n_estimators': 100, 
            'learning_rate': 0.05,
            'max_depth': 5, 
            'random_state': 42,
            'n_jobs': -1
        }
        if model_params:
            default_params.update(model_params)
            
        self.model = XGBRegressor(**default_params)

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train_and_evaluate(self, window=20, threshold=125):
        train_df, test_df, act_df = self.feature_engineer.load_data()
        
        X_train, y_train, X_test, y_test = self.feature_engineer.test_train_split(
            train_df, test_df, act_df, threshold=threshold, window=window
        )
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        print(f"Training XGBRegressor (Window Size: {window}, Target Clip: {threshold})...")
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, predictions)
        print(f'Pipeline Complete! Test RMSE: {rmse:.2f} cycles')
        
        return y_test, predictions
    