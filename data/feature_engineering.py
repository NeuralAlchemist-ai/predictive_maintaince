import pandas as pd
import numpy as np

class Feature_preprocess:
    def __init__(self,DATA_PATH):
        self.data_path = DATA_PATH
        self.columns = [
                        'unit_number', 'cycle', 'setting1', 'setting2', 'setting3',
                        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
                        ]
    def load_data(self):
        train_df = pd.read_csv(self.data_path + "/train_FD001.txt", sep='\s+', header=None, names=self.columns)
        test_df = pd.read_csv(self.data_path + "/test_FD001.txt", sep='\s+', header=None, names=self.columns)
        act_df = pd.read_csv(self.data_path + "/RUL_FD001.txt", sep='\s+', header=None)
        return train_df, test_df, act_df

    def calculate_rul(self, df, threshold):
        df = df.copy()
        df['max_cycle'] = df.groupby('unit_number')['cycle'].transform('max')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df['RUL'] = df['RUL'].clip(upper=threshold)
        return df
    
    def feature_engineering(self, df, window):
        sensor_cols = [col for col in df.columns if col.startswith('s') and col not in ['setting1', 'setting2', 'setting3']]
        processed_groups=[]

        for id, group in df.groupby('unit_number'):
            group=group.sort_values('cycle')
            for col in sensor_cols:
                #Rolling mean
                group[f'{col}_rolling_mean'] = group[col].rolling(window, min_periods=1).mean()
                #Rolling std
                group[f'{col}_rolling_std'] = group[col].rolling(window, min_periods=1).std().bfill()
                #Lag
                group[f'{col}_lag'] = group[col].shift(window).bfill()
                #Velocity
                group[f'{col}_velocity'] = group[col] - group[f'{col}_rolling_mean']
            processed_groups.append(group)
        return pd.concat(processed_groups, ignore_index=True)
    
    def test_train_split(self, train_df, test_df, act_df,threshold, window):
        train_df=self.calculate_rul(train_df, threshold)


        train_processed=self.feature_engineering(train_df, window)
        test_preprocessed=self.feature_engineering(test_df, window)
        
        X_train=train_processed.drop(columns=['unit_number','cycle','max_cycle','RUL'],errors='ignore')
        y_train=train_processed['RUL']

        X_test=test_preprocessed.drop_duplicates(subset='unit_number', keep='last').reset_index(drop=True).drop(columns=['unit_number','cycle'],errors='ignore')

        y_test=np.squeeze(act_df)

        return X_train, y_train, X_test, y_test