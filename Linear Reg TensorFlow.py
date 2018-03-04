import  math

from matplotlib import  cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from sklearn import metrics

import  tensorflow as tf

from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv",sep=',')

dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
dataframe["median_house_value"] /= 1000.0

print(dataframe.describe())

my_feature = dataframe[["total_rooms"]]

feature_columns = [tf.feature_column.numeric_column("total_rooms")]

targets = dataframe["median_house_value"]

my_optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)


linear_reg = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer= my_optimizer
)

