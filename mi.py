import os.path

import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import numpy as np


data_file = "data.csv"
if not os.path.exists(data_file):
    x1 = np.random.uniform(-1, 1, 100)
    x5 = np.random.uniform(-1, 1, 100)
    data = pd.DataFrame({"x1": x1, "x2": x1, "x3": 2*x1, "x4": x1*x1, "x5": x5})
    data.to_csv(data_file, index=False)

dataset = pd.read_csv(data_file).to_numpy()

B1 = dataset[:, 0].reshape(-1, 1)
B2 = dataset[:, 1].reshape(-1, 1)
B3 = dataset[:, 2].reshape(-1, 1)
B4 = dataset[:, 3].reshape(-1, 1)
B5 = dataset[:, 4].reshape(-1, 1)

print(mutual_info_regression(B1, B1.ravel()))
print(mutual_info_regression(B1, B2.ravel()))
print(mutual_info_regression(B1, B3.ravel()))
print(mutual_info_regression(B1, B4.ravel()))
print(mutual_info_regression(B1, B5.ravel()))