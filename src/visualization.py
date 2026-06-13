import matplotlib.pyplot as plt

class Visualizations:
    
    @staticmethod
    def plot_parity(y_test, y_pred):
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title(f'Actual vs Predicted')
        plt.show()