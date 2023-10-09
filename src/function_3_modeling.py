#import all realated libraries
#import libraries for data analysis
import numpy as np
import pandas as pd
# import pickle and json file for columns and model file
import pickle
import json
import joblib
import yaml
import scipy.stats as scs
# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import uuid
from tqdm import tqdm
import pandas as pd
import os
import copy
import util as util
from sklearn.model_selection import cross_val_score
from datetime import datetime
import numpy as np

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import uuid
from datetime import datetime
import numpy as np
###################################################

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

def load_data_scaling(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    #Read data X_train dan y_sm hasil dari data processing
    X_train = util.pickle_load(config_data["robust_scaler_train"][0])
    y_train = util.pickle_load(config_data["robust_scaler_train"][1])

    #Read data X_valid dan y_valid hasil dari data preparation
    X_valid = util.pickle_load(config_data["robust_scaler_valid"][0])
    y_valid = util.pickle_load(config_data["robust_scaler_valid"][1])

    #Read data X_test dan y_test hasil dari data preparation
    X_test = util.pickle_load(config_data["robust_scaler_test"][0])
    y_test = util.pickle_load(config_data["robust_scaler_test"][1])

    # Return 3 set of data
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_xgboost_with_grid_search(X_train, y_train, X_test, y_test, X_valid, y_valid):
    # Membangun objek XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(random_state=42)

    # Daftar parameter yang ingin dioptimalkan
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'n_estimators': [10, 50, 100],
    }

    # Membangun objek GridSearchCV dengan XGBRegressor dan parameter grid
    grid_search_xgb = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_absolute_error', refit=True)

    # Melatih model dengan Grid Search
    grid_search_xgb.fit(X_train, y_train)

    # Mendapatkan model dengan parameter terbaik
    best_xgb_model = grid_search_xgb.best_estimator_

    # Menampilkan parameter terbaik
    print("Parameter Terbaik:")
    print(grid_search_xgb.best_params_)

    # Melakukan refit model terbaik ke data pelatihan
    best_xgb_model.fit(X_train, y_train)

    # Predict
    y_pred_train_xgb = best_xgb_model.predict(X_train)
    y_pred_test_xgb = best_xgb_model.predict(X_test)
    y_pred_valid_xgb = best_xgb_model.predict(X_valid)

    # MAE
    mae_train_cvxgb = mean_absolute_error(y_train, y_pred_train_xgb)
    mae_test_cvxgb = mean_absolute_error(y_test, y_pred_test_xgb)
    mae_valid_cvxgb = mean_absolute_error(y_valid, y_pred_valid_xgb)

    # R2
    r2_train_cvxgb = r2_score(y_train, y_pred_train_xgb)
    r2_test_cvxgb = r2_score(y_test, y_pred_test_xgb)
    r2_valid_cvxgb = r2_score(y_valid, y_pred_valid_xgb)

    # MSE
    mse_train_cvxgb = mean_squared_error(y_train, y_pred_train_xgb)
    mse_test_cvxgb = mean_squared_error(y_test, y_pred_test_xgb)
    mse_valid_cvxgb = mean_squared_error(y_valid, y_pred_valid_xgb)

    # RMSE
    rmse_train_cvxgb = np.sqrt(mse_train_cvxgb)
    rmse_test_cvxgb = np.sqrt(mse_test_cvxgb)
    rmse_valid_cvxgb = np.sqrt(mse_valid_cvxgb)

    # Menampilkan metrik evaluasi
    print("MAE (Mean Absolute Error) Train: ", mae_train_cvxgb)
    print("MAE (Mean Absolute Error) Test: ", mae_test_cvxgb)
    print("MAE (Mean Absolute Error) Valid: ", mae_valid_cvxgb)

    print("R2 (R-squared) Train: ", r2_train_cvxgb)
    print("R2 (R-squared) Test: ", r2_test_cvxgb)
    print("R2 (R-squared) Valid: ", r2_valid_cvxgb)

    print("MSE (Mean Squared Error) Train: ", mse_train_cvxgb)
    print("MSE (Mean Squared Error) Test: ", mse_test_cvxgb)
    print("MSE (Mean Squared Error) Valid: ", mse_valid_cvxgb)

    print("RMSE (Root Mean Squared Error) Train: ", rmse_train_cvxgb)
    print("RMSE (Root Mean Squared Error) Test: ", rmse_test_cvxgb)
    print("RMSE (Root Mean Squared Error) Valid: ", rmse_valid_cvxgb)
    
    return best_xgb_model

def save_training_log(model, model_name, X_train, y_train, X_test, y_test, X_valid, y_valid):
    # Generate unique id
    model_uid = uuid.uuid4().hex
    
    # Get current time and date
    now = datetime.now()
    training_time = now.strftime("%H:%M:%S")
    training_date = now.strftime("%Y-%m-%d")
    
    best_params_xgb = model.get_params()
    
    # Filter parameter yang ingin Anda tampilkan (misalnya, hanya yang berhubungan dengan tuning)
    selected_params = {
    'colsample_bytree': best_params_xgb['colsample_bytree'],
    'max_depth': best_params_xgb['max_depth'] if best_params_xgb['max_depth'] is None else "None",
    'min_child_weight': best_params_xgb['min_child_weight'],
    'n_estimators': best_params_xgb['n_estimators'],
    'subsample': best_params_xgb['subsample']
    }
    
    # Predict
    y_pred_train_xgb = model.predict(X_train)
    y_pred_test_xgb = model.predict(X_test)
    y_pred_valid_xgb = model.predict(X_valid)

    # MAE
    mae_train_cvxgb = mean_absolute_error(y_train, y_pred_train_xgb)
    mae_test_cvxgb = mean_absolute_error(y_test, y_pred_test_xgb)
    mae_valid_cvxgb = mean_absolute_error(y_valid, y_pred_valid_xgb)

    # R2
    r2_train_cvxgb = r2_score(y_train, y_pred_train_xgb)
    r2_test_cvxgb = r2_score(y_test, y_pred_test_xgb)
    r2_valid_cvxgb = r2_score(y_valid, y_pred_valid_xgb)

    # MSE
    mse_train_cvxgb = mean_squared_error(y_train, y_pred_train_xgb)
    mse_test_cvxgb = mean_squared_error(y_test, y_pred_test_xgb)
    mse_valid_cvxgb = mean_squared_error(y_valid, y_pred_valid_xgb)

    # RMSE
    rmse_train_cvxgb = np.sqrt(mse_train_cvxgb)
    rmse_test_cvxgb = np.sqrt(mse_test_cvxgb)
    rmse_valid_cvxgb = np.sqrt(mse_valid_cvxgb)

    # Create a log dictionary
    log = {
        "model_name": model_name,
        "model_uid": model_uid,
        "training_time": training_time,
        "training_date": training_date,
        "best_params": selected_params,
        "MAE_train": mae_train_cvxgb,
        "MAE_test": mae_test_cvxgb,
        "MAE_valid": mae_valid_cvxgb,
        "R2_train": r2_train_cvxgb,
        "R2_test": r2_test_cvxgb,
        "R2_valid": r2_valid_cvxgb,
        "MSE_train": mse_train_cvxgb,
        "MSE_test": mse_test_cvxgb,
        "MSE_valid": mse_valid_cvxgb,
        "RMSE_train": rmse_train_cvxgb,
        "RMSE_test": rmse_test_cvxgb,
        "RMSE_valid": rmse_valid_cvxgb
    }
    
    # Save log as a JSON file
    log_file_path = 'training_log/training_log.json'
    with open(log_file_path, 'w') as f:
        json.dump(log, f)

        
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()
    
    # 2. Load dataset
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_scaling(config_data)
    
    # Panggil fungsi untuk melatih model
    best_model_xgb = train_xgboost_with_grid_search(X_train, y_train, X_test, y_test, X_valid, y_valid)  # Store the result in grid_search_xgb
    
    # Panggil fungsi untuk menyimpan log pelatihan
    save_training_log(model_name="XGBoostRegressorCV", model=best_model_xgb, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                      X_valid=X_valid, y_valid=y_valid)
    
    #Save Model
    xgboost_cv = config_data["model_final"]
    with open(xgboost_cv, 'wb') as file:
        pickle.dump(best_model_xgb, file)
