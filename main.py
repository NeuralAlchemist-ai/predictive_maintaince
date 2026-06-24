from config import PipelineConfig
from src.model import PredictiveModel
from src.tuner import HyperparameterTuner
import mlflow
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run the predictive maintenance pipeline.')
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='FD001', 
        help='Dataset name (default: FD001)'
    )
    
    return parser.parse_args()

args =parse_args()
DATASET_NAME = args.dataset
config = PipelineConfig(dataset_name=DATASET_NAME)
DATA_PATH = config.data_path
URL = config.url

mlflow.set_tracking_uri(URL)
mlflow.set_experiment(f"CMAPSS_{DATASET_NAME}")

tuner =  HyperparameterTuner(config)

with mlflow.start_run(run_name='Tuning with Optuna'):
    mlflow.set_tags({"framework": "XGBoost", "phase": "tuning", "dataset": DATASET_NAME})
    best_params = tuner.tune(n_trials=20)

    xgb_keys = ['n_estimators', 'learning_rate', 'max_depth', 'subsample']
    best_model_params = {k: best_params[k] for k in xgb_keys if k in best_params}
    best_window = best_params.get('window', 20)
    best_threshold = best_params.get('threshold', 125)

with mlflow.start_run(run_name='Production Model'):
    mlflow.set_tags({"framework": "XGBoost", "phase": "production", "dataset": DATASET_NAME})
    mlflow.log_params(best_params)
        
    final_pipeline = PredictiveModel(config, model_params=best_model_params)
    y_test, y_pred, test_rmse = final_pipeline.train_and_evaluate(window=best_window, threshold=best_threshold)
    mlflow.log_metric('Test_RMSE', test_rmse)
    mlflow.xgboost.log_model(final_pipeline.model, artifact_path="model", conda_env="custom_env.yaml", registered_model_name=f"XGBRegressor_{DATASET_NAME}")


