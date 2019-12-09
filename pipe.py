import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tempfile import TemporaryFile

import validators
import csv
import cv2
import urllib

data_uk = pd.DataFrame(pd.read_csv("./data/GBvideos.csv", error_bad_lines=False))
data_us = pd.DataFrame(pd.read_csv("./data/USvideos.csv", error_bad_lines=False))
data = pd.concat([data_us, data_uk])

print("number of rows in data: " + str(data.shape[0]))

likes = data[['likes', 'dislikes', 'comment_total']]
views = data[['views']]

data_dmatrix = xgb.DMatrix(data=likes,label=views)

likes_train, likes_test, views_train, views_test = train_test_split(likes, views, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(likes_train,views_train)

preds = xg_reg.predict(likes_test)

rmse = np.sqrt(mean_squared_error(views_test, preds))
print("RMSE: %f" % (rmse))

# Visualize feature importance
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()