from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class RegimeBasedScaler(BaseEstimator, TransformerMixin): 
    def __init__(self, config):
        self.config = config
        self.dataset_name = self.config.dataset_name
        self.cluster_model_class = self.config.cluster_model_class
        self.random_state = self.config.random_state
        self.num_clusters = self.config.num_clusters
        self.scaler_class = self.config.scaler_class

        self.cluster_model = None
        self.setting_cols = ['setting1', 'setting2', 'setting3']
        
        self.scalers_ = {} 

    def fit(self, X, y=None):
        self.cluster_model = self.cluster_model_class(n_clusters=self.num_clusters, random_state=self.random_state)
        
        labels = self.cluster_model.fit_predict(X[self.setting_cols])
        
        self.scalers_ = {}
        for cluster in range(self.num_clusters):
            mask = labels == cluster
            x_cluster = X[mask].drop(columns=self.setting_cols)

            scaler = self.scaler_class() 
            
            if not x_cluster.empty: 
                scaler.fit(x_cluster)

            self.scalers_[cluster] = scaler

        return self
        
    def transform(self, X):
        X_res = X.copy()
        labels = self.cluster_model.predict(X_res[self.setting_cols])
        X_res['cluster_label'] = labels

        sensor_cols = [
                c for c in X_res.columns if c not in self.setting_cols + ["cluster_label"]
            ]
        X_res[sensor_cols] = X_res[sensor_cols].astype(float)

        for cluster in range(self.num_clusters):
            mask = labels == cluster

            if not X_res[mask].empty:
                X_sensors = X_res.loc[mask, sensor_cols]

                if cluster in self.scalers_:
                    scaled_data = self.scalers_[cluster].transform(X_sensors)
                    X_res.loc[mask, sensor_cols] = scaled_data
                    
        return X_res
        
class TimeSeriesFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, config, window):
        self.config = config
        self.window = window
        self.sensor_cols = self.config.sensor_cols 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        processed_groups = []

        for id, group in X.groupby('unit_number'):
            group = group.sort_values('cycle')
            for col in self.sensor_cols:
                group[f'{col}_rolling_mean'] = group[col].rolling(self.window, min_periods=1).mean()
                group[f'{col}_rolling_std'] = group[col].rolling(self.window, min_periods=1).std().bfill()
                group[f'{col}_lag'] = group[col].shift(self.window).bfill()
                group[f'{col}_velocity'] = group[col] - group[f'{col}_rolling_mean']
            processed_groups.append(group)
            
        return pd.concat(processed_groups, ignore_index=True)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop = [
                "unit_number",
                "cycle",
                "setting1",
                "setting2",
                "setting3",
                "cluster_label",
            ]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_filtered = X.copy()

        existing_cols_to_drop = [
            col for col in self.cols_to_drop if col in X_filtered.columns
        ]

        return X_filtered.drop(columns=existing_cols_to_drop)