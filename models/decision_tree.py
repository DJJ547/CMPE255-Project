import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

ROOT_PATH = os.path.dirname(os.getcwd())
DIRECTORY_PATH = 'CMPE255-Project/static'
DATASET_NAME = 'diabetes_binary_health_indicators_BRFSS2015.csv'
DATASET_PATH = os.path.join(ROOT_PATH, DIRECTORY_PATH, DATASET_NAME)

def preprocess_data():
    print("dataset path: ", DATASET_PATH)
    # read in the dataset
    df = pd.read_csv(filepath_or_buffer=DATASET_PATH)
    print(df)
    # Convert dataframe to numpy array
    df = df.to_numpy()
    print(df.shape)
    print(df)
    # retrieve the first label column
    label_df = df[:, 0]
    print(label_df.shape)
    print(label_df)
    # retrieve dataframe without label column
    data_df = df[:, 1:]
    print(data_df.shape)
    print(data_df)
    # Split numpy array into random train and validation subsets
    X_train, X_test, y_train, y_test = train_test_split(data_df, label_df, test_size=0.2, random_state=198,
                                                        shuffle=True)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test

def get_column_names():
    df = pd.read_csv(filepath_or_buffer=DATASET_PATH)
    column_names = df.columns.values.tolist()
    return column_names

def train_model(X_train, y_train):
    X_train, X_test, y_train, y_test = preprocess_data()
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt


def evaluate_model(dt, X_train, y_train, X_test, y_test, col_names):
    class_names = ['diabetic', 'non-diabetic']
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(dt,
                       feature_names=col_names,
                       class_names=class_names,
                       filled=True)

    # the score, or accuracy of the model
    dt.score(X_test, y_test)

    scores = cross_val_score(dt, X_train, y_train, cv=10)
    print(np.mean(scores))

    predictions = dt.predict(X_test)
    print(classification_report(y_test, predictions))


def predict_model(input):
    X_train, X_test, y_train, y_test = preprocess_data()
    dt = train_model(X_train, y_train)
    return dt.predict(input)

# if __name__=="__main__":
#     train_decision_tree_model()
