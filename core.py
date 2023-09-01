import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#Устанавливаем стиль для графиков

sns.set(style="whitegrid", font_scale=1.3)
matplotlib.rcParams["legend.framealpha"] = 1
matplotlib.rcParams["legend.frameon"] = True

#Фиксируем генератор случайных чисел для воспроизводимости эксперимента
np.random.seed(42)
torch.manual_seed(42);
df = pd.read_csv('Boston.csv')
df.head()
y = df['medv']
X = df[['crim', 'zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']]
X.head()

y[:5]

X.describe()

#Также посмотрим на матрицу корреляций между признаками
corr = X.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18, 18))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap,
            square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax);

#Разобьем наши данные на обучающую и валидационную выборки в пропорции  41
plt.figure(figsize=(18, 8))
plt.subplot(121)
plt.scatter(X_train.rm, y_train, label="Train")
plt.scatter(X_val.rm, y_val, c="r", label="Validation")
plt.xlabel("Average number of rooms per dwelling")
plt.ylabel("Price, $")
plt.legend(loc="lower right", frameon=True)
plt.subplot(122)
plt.scatter(X_train.rad, y_train, label="Train")
plt.scatter(X_val.rad, y_val, c="r", label="Validation")
plt.xlabel("Index of accessibility to radial highways")
plt.ylabel("Price, $")
plt.legend(loc="lower right");

#Сделаем нормализацию каждого признака в диапазон  (0;1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)