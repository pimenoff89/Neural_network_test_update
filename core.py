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