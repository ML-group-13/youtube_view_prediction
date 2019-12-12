import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

def read_in_data():
	data_uk = pd.DataFrame(pd.read_csv("./data/GBvideos.csv", error_bad_lines=False))
	data_us = pd.DataFrame(pd.read_csv("./data/USvideos.csv", error_bad_lines=False))
	data = pd.concat([data_us, data_uk])

	data['images'] = read_images()
	data = data.loc[data['images'] != '']

	print("number of rows in cleaned data: " + str(data.shape[0]))
	return data

def read_images():
	images_data = []

	for file in [f for f in listdir("./data/images") if isfile(join("./data/images", f))]:
		images = np.load("./data/images/" + file, allow_pickle=True)
		for image in images['arr_0']:
			images_data.append(image)

if __name__ == "__main__":
	data = read_in_data()
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