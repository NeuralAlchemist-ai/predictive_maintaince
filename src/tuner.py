import optuna
from src.model import PredictiveModel
from src.utils import prepare_data
import mlflow

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.random_state = config.random_state
        self.raw_train_df = None

    def objective(self, trial):
        model_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': self.random_state,
        'n_jobs': -1
        }

        window = trial.suggest_int('window', 10, 30)
        threshold = trial.suggest_int('threshold', 110, 140)
        
        with mlflow.start_run(run_name=f"Optuna Trial {trial.number}", nested=True):
            mlflow.log_params(model_params)
            mlflow.log_param('window', window)
            mlflow.log_param('threshold', threshold)


            X_train, y_train = prepare_data(self.raw_train_df, threshold)
            pipeline = PredictiveModel(self.config, model_params=model_params)
            cv_rmse = pipeline.train_cv(X_train, y_train, window=window)

            mlflow.log_metric('CV_RMSE', cv_rmse)
    
        return cv_rmse

    def tune(self, raw_train_df, n_trials):
        self.raw_train_df = raw_train_df

        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params