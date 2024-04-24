# Import libraries
import argparse
import glob
import os
import pandas as pd
import numpy as np
import logging
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Create a logger


# define functions
def main(args):

    # Enable mlflow autologging
    mlflow.sklearn.autolog()
    
     # Enable logging
    logger.info("Logging enabled.")
    
    # Read data
    df = get_csvs_df(args.training_data)


    # split data
    X_train, X_test, y_train, y_test = split_data(df, 
                                                  ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 
                                                   'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age'], 
                                                  'Diabetic')

    # Train model
    model = train_model(args.reg_rate, X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.2f}")


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data - Done
def split_data(df, features, target, test_size=0.3, random_state=0):
    X = df[features].values
    y = df[target].values
    print(np.unique(y, return_counts=True))  # Optional: print unique values of target variable with their counts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # Log a message to indicate the start of the script
    logger.info("\n\n")
    logger.info("*" * 60)

    # Parse arguments
    args = parse_args()

    # Run main function
    main(args)

    # Log a message to indicate the end of the script
    logger.info("*" * 60)
    logger.info("\n\n")
