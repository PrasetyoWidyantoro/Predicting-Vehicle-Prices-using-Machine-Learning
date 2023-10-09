from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import joblib
import yaml
from datetime import datetime
import util as util
import numpy as np
# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")
# import pickle and json file for columns and model file
import pickle
import json
import copy

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        if i.endswith(".csv"):  # Hanya membaca file CSV
            raw_dataset = pd.concat([pd.read_csv(os.path.join(raw_dataset_dir, i)), raw_dataset])
            
    # Return raw dataset
    return raw_dataset

def check_car_data(input_data, params):
    # Check data types
    assert input_data.select_dtypes("float").columns.to_list() == params["float64_columns"], "Error in float64 columns."
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "Error in object columns."

    # Check range of data
    assert input_data.Year.between(params["range_Year"][0], params["range_Year"][1]).sum() == len(input_data), "Error in Year range."
    assert input_data.Present_Price.between(params["range_Present_Price"][0], params["range_Present_Price"][1]).sum() == len(input_data), "Error in Present_Price range."
    assert input_data.Kms_Driven.between(params["range_Kms_Driven"][0], params["range_Kms_Driven"][1]).sum() == len(input_data), "Error in Kms_Driven range."

    assert set(input_data.Fuel_Type).issubset(set(params["range_Fuel_Type"])), "Error in Fuel_Type range."
    assert set(input_data.Seller_Type).issubset(set(params["range_Seller_Type"])), "Error in Seller_Type range."
    assert set(input_data.Transmission).issubset(set(params["range_Transmission"])), "Error in Transmission range."

    assert set(input_data.Owner).issubset(set(params["range_Owner"])), "Error in Owner categories."

######################################################################################################################    

if __name__ == "__main__":
    #1. Load configuration file
    config_data = util.load_config()
    
    # 2.Read all raw Dataset
    vehicles_data = read_raw_data(config_data).drop('Car_Name', axis=1) 
        
    #4. Reset index
    vehicles_data.reset_index(
        inplace = True,
        drop = True
    )

    #5. Splitting input output
    X = vehicles_data.drop(columns = "Selling_Price")
    y = vehicles_data["Selling_Price"]

    #6. Split Data 75% training 25% testing
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y,
        test_size = 0.25,
        random_state = 123)
    
    #7. Split data train menjadi train dan validation set
    X_test, X_valid, \
    y_test, y_valid = train_test_split(
        X_test, y_test, 
        test_size=0.4, 
        random_state=123)
    
    #Menggabungkan x train dan y train untuk keperluan EDA
    util.pickle_dump(X_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(X_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(X_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])
    
    print("Data Pipeline passed successfully.")
