import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


class LightGBMRegressor:
    def __init__(self):
        self.params = None
        self.model = None
        self.metrics = {}
    
    
    def train(self, X_train, y_train):
        train_data = lgb.Dataset(X_train, label=y_train)
        

        self.model = lgb.train(self.params, 
                                train_data)
        
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Run 'train' method first.")
        return self.model.predict(X)
    
    def validate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained. Run 'train' method first.")
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        self.metrics = {"MAE": mae, 
                        "RMSE": rmse}