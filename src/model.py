from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from data.feature_engineering import Feature_preprocess

class PredictiveModel:
    def __init__(self, DATA_PATH):
        self.data_path = DATA_PATH
        self.feature_engineer = Feature_preprocess(DATA_PATH)
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    def train(self):
        train_df, test_df, act_df = self.feature_engineer.load_data()
        X_train, y_train, X_test, y_test = self.feature_engineer.test_train_split(train_df, test_df, act_df)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error: {mse}')