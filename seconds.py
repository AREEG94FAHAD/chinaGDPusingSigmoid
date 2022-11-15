import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv("dataset.csv")
# dataset.head(10)


def sigmoid(x, Beta_1, Beta_2):
      y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
      return y


x_data, y_data = (dataset["Year"].values, dataset["Value"].values)

# Lets normalize our data
xdata =x_data / max(x_data)
ydata =y_data / max(y_data)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)




# split data into train/test

msk = np.random.rand(len(dataset)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]



popt, pcov = curve_fit(sigmoid, train_x, train_y)

y_hat = sigmoid(test_x, *popt)

from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )








