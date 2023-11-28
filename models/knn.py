import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error , accuracy_score
from mlxtend.plotting import plot_confusion_matrix

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


def train_model(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    return knn


def evaluate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    print("predicted: ", y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error : ' + str(mse))
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error : ' + str(rmse))

    matrix = classification_report(y_test, y_pred)
    print(matrix)

    # cm1 = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(conf_mat=cm1, show_absolute=True, show_normed=True, colorbar=True)
    # plt.show()


def predict_model(input):
    X_train, X_test, y_train, y_test = preprocess_data()
    knn = train_model(X_train, y_train)
    return knn.predict(input)


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = preprocess_data()
#     knn_model = train_knn_model(X_train, y_train)
#     evaluate_knn_model(knn_model, X_test, y_test)
