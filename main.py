import argparse
import mlflow
from config import PipelineConfig
from src.model import PredictiveModel
from src.tuner import HyperparameterTuner
from src.utils import load_data, prepare_data

def parse_args():
    parser = argparse.ArgumentParser(description='Run the predictive maintenance pipeline.')
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset name (default: FD001)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials for Tuner(default: 20)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    DATASET_NAME = args.dataset
    config = PipelineConfig(dataset_name=DATASET_NAME)
    DATA_PATH = config.data_path
    URL = config.url
    N_TRIALS = args.n_trials
    
    mlflow.set_tracking_uri(URL)
    mlflow.set_experiment(f"CMAPSS_{DATASET_NAME}")

    raw_train_df, X_test, act_df = load_data(DATA_PATH, DATASET_NAME)
    tuner = HyperparameterTuner(config)

    with mlflow.start_run(run_name='Tuning with Optuna'):
        mlflow.set_tags({"framework": "XGBoost", "phase": "tuning", "dataset": DATASET_NAME})
        best_params = tuner.tune(raw_train_df, n_trials=N_TRIALS)

        best_window = best_params.get('window', 20)
        best_threshold = best_params.get('threshold', 125)

        best_model_params = {k: v for k, v in best_params.items() if k not in ['window', 'threshold']}
           
    with mlflow.start_run(run_name='Production Model'):
        mlflow.set_tags({"framework": "Scikit-Learn", "phase": "production", "dataset": DATASET_NAME})
        mlflow.log_params(best_params)
            
        X_train, y_train = prepare_data(raw_train_df, threshold=best_threshold)
        
        y_test = act_df.clip(upper=best_threshold).values.ravel()

        final_pipeline = PredictiveModel(config, model_params=best_model_params)
        
        y_test_actual, y_pred, test_rmse = final_pipeline.train_and_evaluate(
            X_train, y_train, X_test, y_test, window=best_window
        )
        
        mlflow.log_metric('Test_RMSE', test_rmse)
        
        mlflow.sklearn.log_model(
            sk_model=final_pipeline.pipeline, 
            artifact_path="model", 
            registered_model_name=f"RUL_Pipeline_{DATASET_NAME}"
        )


