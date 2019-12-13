import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import sys


def read_in_data(development=False):
    print("started reading in data")
    data_uk = pd.DataFrame(pd.read_csv(
        "./data/GBvideos.csv", error_bad_lines=False))
    data_us = pd.DataFrame(pd.read_csv(
        "./data/USvideos.csv", error_bad_lines=False))
    data = pd.concat([data_us, data_uk])

    if (development):
        data = data[:10000]

    data['images'] = read_images(development)
    data = data.loc[data['images'] != '']

    print("number of rows in cleaned data: " + str(data.shape[0]))
    return data


def read_images(development=False):
    print("started reading in images, this may take a few minutes")
    images_data = []
    for file in [f for f in listdir("./data/images") if isfile(join("./data/images", f))]:
        if development and len(file.split("_")[1]) > 5:
            continue
        images = np.load("./data/images/" + file, allow_pickle=True)
        for image in images['arr_0']:
            images_data.append(image)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "development":
            print("running in development mode")
            data = read_in_data(development=True)
        else:
            data = read_in_data()
    else:
        data = read_in_data()

    features = data[['likes', 'dislikes', 'comment_count']]
    target = data[['views']]

    data_dmatrix = xgb.DMatrix(data=features, label=target)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=123)

    xgbr = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                            max_depth=5, alpha=10, n_estimators=10)

    xgbr.fit(x_train, y_train)

    scores = cross_val_score(xgbr, x_train, y_train, cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())

    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(xgbr, x_train, y_train, cv=kfold)
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

    predictions = xgbr.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % np.sqrt(mse))

    # Visualize feature importance
    xgb.plot_importance(xgbr)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
