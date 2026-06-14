import optuna
from src.model import PredictiveModel
import mlflow

class HyperparameterTuner:
    def __init__(self, data_path):
        self.data_path = data_path

    def objective(self, trial):
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }

        window = trial.suggest_int('window', 10, 30)
        threshold = trial.suggest_int('threshold', 110, 140)
        
        with mlflow.start_run(run_name=f"Optuna Trial {trial.number}"):
            mlflow.log_params(model_params)
            mlflow.log_param('window', window)
            mlflow.log_param('threshold', threshold)
        
            pipeline = PredictiveModel(self.data_path, model_params=model_params)
            cv_rmse = pipeline.train_cv(window=window, threshold=threshold)

            mlflow.log_metric('CV_RMSE', cv_rmse)
    
        return cv_rmse

    def tune(self, n_trials):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params