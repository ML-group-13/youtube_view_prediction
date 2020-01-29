import sys
import time
import csv
import warnings
from os import listdir
from os.path import isfile, join
from progress.bar import ChargingBar
from statistics import mean

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from numpy import sort

from feature_extraction.text_feature_extraction.text_feature_extractor import TextFeatureExtractor
from feature_extraction.image_feature_extraction.image_feature_extractor import ImageFeatureExtractor


# WARNINGS
warnings.simplefilter(action='ignore', category=FutureWarning)


def read_in_data(development=False, datapoints=10000):
    data_uk = pd.DataFrame(pd.read_csv(
        "./data/GBvideos.csv", error_bad_lines=False))
    data_us = pd.DataFrame(pd.read_csv(
        "./data/USvideos.csv", error_bad_lines=False))
    data = pd.concat([data_us, data_uk])

    if (development):
        data = data[:datapoints]

    data['images'] = read_images(development, datapoints)
    data = data.loc[data['images'] != '']

    return data


def read_images(development=False, datapoints=10000):
    images_data = []

    for file in [f for f in listdir("./data/imagesHD") if (isfile(join("./data/imagesHD", f)))]:
        if (int(file.split("_")[1]) < datapoints):
            if development and int(file.split("_")[1]) > datapoints:
                continue
            images = np.load("./data/imagesHD/" + file, allow_pickle=True)
            bar = ChargingBar('Reading ' + file + ':\t',
                              max=len(images['arr_0']))
            for image in images['arr_0']:
                bar.next()
                images_data.append(image)
            bar.finish()
        else:
            break

    images_data = images_data[:datapoints]
    return images_data


def k_fold_test(features, target, xgbr_model, k=5):
    train_rmse = []
    test_rmse = []
    train_rmse_percent = []
    test_rmse_percent = []

    for i in range(0, k):
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.25)
        xgbr_model.fit(x_train, y_train)

        predictions_train = xgbr_model.predict(x_train)
        mse = mean_squared_error(y_train, predictions_train)
        train_rmse_percent.append(np.sqrt(mse) / mean(target['views']) * 100)
        train_rmse.append(np.sqrt(mse))

        predictions_test = xgbr_model.predict(x_test)
        mse = mean_squared_error(y_test, predictions_test)
        test_rmse_percent.append(np.sqrt(mse) / mean(target['views']) * 100)
        test_rmse.append(np.sqrt(mse))

    return ((mean(train_rmse), mean(train_rmse_percent)), (mean(test_rmse), mean(test_rmse_percent)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "development":
            # print("running in development mode")
            if len(sys.argv) == 3:
                try:
                    data = read_in_data(
                        development=True, datapoints=int(sys.argv[2])
                    )
                except ValueError:
                    data = read_in_data(development=True)
            else:
                data = read_in_data(development=True)
        else:
            data = read_in_data()
    else:
        data = read_in_data()

    start_time = int(time.time())
    # TODO: Maybe change the y_variable to log_views instead of log_views
    # data['log_views'] = np.log(data['views'])

    # data['views'].plot.hist(bins=100, alpha=0.5)
    # data['log_views'].plot.hist(bins=100, alpha=0.5)

    data = TextFeatureExtractor().extract_features(data)
    data = ImageFeatureExtractor().extractImageFeatures(data)
    text_features_partly = ['title_sentiment', 'title_sentiment_polarity',
                           'title_length', 'title_capitals_count',
                           'title_capitals_ratio', 'title_non_letter_count', 'title_non_letter_ratio',
                           'title_word_count', 'title_number_count']
    image_features_partly = ['image_amount_of_faces', 'image_text',
                            'image_emotions_angry', 'image_emotions_sad', 'image_emotions_neutral',
                            'image_emotions_disgust', 'image_emotions_surprise', 'image_emotions_fear',
                            'image_emotions_happy']
    general_features_partly = ['category_id']

    total_features_names = text_features_partly + \
        image_features_partly + general_features_partly
    text_features_names = text_features_partly + general_features_partly
    image_features_names = image_features_partly + general_features_partly

    total_features = data[total_features_names]
    text_features = data[text_features_names]
    image_features = data[image_features_names]

    features = [text_features, image_features]
    names_per_feature = [total_features_names,
    text_features_names, image_features_names]
    feature_names = ["text features", "image features"]
    target = data[['views']]

    file = open('./results/results_' + str(int(time.time())) + '.csv', 'w')
    with file:
        writer = csv.writer(file)
        writer.writerow(["model name", "depth", "n_estimator", "colsample_bytree", "learning rate", "alpha", "features"
                         "train_rmse", "train_rmse_percent", "test_rmse", "test_rmse_percent"])
        for i in range(0, len(features)):
            print("\nFitting model with " + feature_names[i])
            # depths = [3, 5, 7, 9, 11]
            depths = [9, 11]
            n_estimators = [10, 50, 100, 250, 400]
            colsamples_bytree = [0.25, 0.5, 0.75]
            learning_rates = [0.01, 0.05, 0.1]
            alphas = [5, 10, 20, 30, 50, 100]
            best_model = (99999999, [])
            for depth in depths:
                for n_estimator in n_estimators:
                    for colsample_bytree in colsamples_bytree:
                        for learning_rate in learning_rates:
                            for alpha in alphas:
                                temp_names_per_feature = list(
                                    names_per_feature[i])
                                feature_data = data[names_per_feature[i]]
                                xgbr_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=colsample_bytree, learning_rate=learning_rate,
                                                              max_depth=depth, alpha=alpha, n_estimators=n_estimator)
                                xgbr_model.fit(feature_data, target)
                                while len(temp_names_per_feature) > 1:
                                    index = np.where(xgbr_model.feature_importances_ == min(
                                        xgbr_model.feature_importances_))[0][0]
                                    del feature_data[temp_names_per_feature[index]]
                                    del temp_names_per_feature[index]
                                    train_results, test_results = k_fold_test(
                                        feature_data, target, xgbr_model)
                                    train_rmse, train_rmse_percent = train_results
                                    test_rmse, test_rmse_percent = test_results
                                    row = [feature_names[i], depth, n_estimator, colsample_bytree, learning_rate, alpha, temp_names_per_feature,
                                           train_rmse, train_rmse_percent, test_rmse, test_rmse_percent]
                                    writer.writerow(row)
                                    print("Configurations   Depth: {}, n_estimators: {}, colsample_bytree: {}, learning rate: {}, alpha: {}, features: {}"
                                          .format(depth, n_estimator, colsample_bytree, learning_rate, alpha, temp_names_per_feature))
                                    print("Train RMSE: {:10.2f}, Test RMSE: {:10.2f}".format(
                                          train_rmse, test_rmse))
                                    print("Train RMSE in %: {:10.2f}%, Test RMSE in %: {:10.2f}%".format(
                                          train_rmse_percent, test_rmse_percent))
                                    if test_rmse < best_model[0]:
                                        best_model = (test_rmse_percent, [
                                                      depth, n_estimator, colsample_bytree, learning_rate, alpha, temp_names_per_feature])
        print("Best configuration with RMSE of: {}    Depth: {:4d}, n_estimators: {:4d}, colsample_bytree: {:10.2f}, learning rate: {:10.2f}, alpha: {:4d}, features: {:4d}"
              .format(best_model[0], best_model[1][0], best_model[1][1], best_model[1][2], best_model[1][3], best_model[1][4], best_model[1][5]))
        print("Running the program took {} seconds ({} minutes)"
            .format(int(time.time) - start_time, int((int(time.time) - start_time) / 60)))
