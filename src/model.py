from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from data.feature_engineering import RegimeBasedScaler, TimeSeriesFeatureGenerator, FeatureSelector

import numpy as np


class PredictiveModel:
    def __init__(self, config, model_params=None):
        self.config = config

        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': config.random_state,
            'n_jobs': -1,
        }
        if model_params:
            default_params.update(model_params)

        self.model = XGBRegressor(**default_params)

        self.pipeline = Pipeline([
            ('timeseries', TimeSeriesFeatureGenerator(config, window=20)),
            ('scaler', RegimeBasedScaler(config)),
            ('selector', FeatureSelector()),
            ('model', self.model),
        ])

    def train_cv(self, X_train, y_train, window=20, n_splits=5):
       
        self.pipeline.set_params(timeseries__window=window)

        groups = X_train['unit_number']
        gkf = GroupKFold(n_splits=n_splits)

        cv_scores = cross_val_score(
            self.pipeline,
            X_train,
            y_train,
            cv=gkf,
            groups=groups,
            scoring='neg_root_mean_squared_error',
        )

        return float(np.mean(-cv_scores))

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, window=20):
        
        self.pipeline.set_params(timeseries__window=window)
        self.pipeline.fit(X_train, y_train)

        last_cycle_mask = X_test.groupby('unit_number').cumcount(ascending=False) == 0

        all_predictions = self.pipeline.predict(X_test)

        y_pred = all_predictions[last_cycle_mask]
        
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f'Pipeline Complete! Test RMSE: {rmse:.2f} cycles')

        return y_test, y_pred, rmse
