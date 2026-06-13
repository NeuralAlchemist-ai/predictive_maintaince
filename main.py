from config import DATA_PATH
from src.model import PredictiveModel
from src.visualization import Visualizations

if __name__ == "__main__":
    pipeline = PredictiveModel(DATA_PATH)

    y_test, y_pred = pipeline.train_and_evaluate(window=20, threshold=125)
    Visualizations.plot_parity(y_test, y_pred)
