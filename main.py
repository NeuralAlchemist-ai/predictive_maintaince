from config import DATA_PATH, URL
from src.model import PredictiveModel
from src.tuner import HyperparameterTuner

import mlflow

mlflow.set_tracking_uri(URL)
mlflow.set_experiment('Optimized XGBOOST Model')

tuner =  HyperparameterTuner(DATA_PATH)
best_params = tuner.tune(n_trials=20)

xgb_keys = ['n_estimators', 'learning_rate', 'max_depth', 'subsample']
best_model_params = {k: best_params[k] for k in xgb_keys if k in best_params}
best_window = best_params.get('window', 20)
best_threshold = best_params.get('threshold', 125)

final_pipeline = PredictiveModel(DATA_PATH, model_params=best_model_params)
y_test, y_pred, test_rmse = final_pipeline.train_and_evaluate(window=best_window, threshold=best_threshold)

with mlflow.start_run(run_name='XGBOOST optimized model'):
    mlflow.log_param('window', best_params['window'])
    mlflow.log_param('threshold', best_params['threshold'])
    mlflow.log_params(final_pipeline.model.get_params())
    mlflow.log_metric('RMSE', test_rmse)
