import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, mean_squared_error
from math import sqrt

__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'GOFUNDME': self.__eval_crowdfunding,
            'INDIEGOGO': self.__eval_crowdfunding,
        }

    def __eval_crowdfunding(self, y_pred, y_true):
        """
        Compute RMSE
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        rmse = sqrt(mean_squared_error(y_pred, y_true))
        mae = np.mean(np.abs(y_pred, y_true))
        mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100

        return [rmse, mae, mape]

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]