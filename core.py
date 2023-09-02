import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

#from sklearn.datasets import load_boston
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

#Создаем тензоры
X_train_tensor = torch.tensor(np.array(X_train_scaled), dtype=torch.float)
X_val_tensor = torch.tensor(np.array(X_val_scaled), dtype=torch.float)
y_train_tensor = torch.tensor(y_train[:, None], dtype=torch.float)
y_val_tensor = torch.tensor(y_val[:, None], dtype=torch.float)

y_train_tensor = torch.tensor(y_train[:, None], dtype=torch.float)
y_val_tensor = torch.tensor(y_val[:, None], dtype=torch.float)
X_val_tensor[12]
tensor([0.5141, 0.0000, 0.6430, 0.0000, 0.6337, 0.1334, 1.0000, 0.0481, 1.0000,
        0.9141, 0.8085, 0.2218, 0.9727])

def mape_loss(input, target):
    return torch.mean(F.l1_loss(input, target, reduction="none") / target) * 100


#Теперь определим функцию ошибки
loss_func = F.mse_loss
#И также зададим набор метрик, которые хотим отслеживать
metrics_func = [loss_func, mape_loss]
metrics_name = ["MSE", "MAPE"]

#Определим функцию для оценки качества одной модели на заданном датасете по заданным метрикам
def evaluate(model, metrics_func, X, y):
    metrics_value = []
    with torch.no_grad():
        preds = model(X)
        for metric_func in metrics_func:
            metric_value = metric_func(torch.FloatTensor(preds).flatten(), torch.FloatTensor(y).flatten())
            metrics_value.append(metric_value)
    return metrics_value


#Утилита, чтобы оценивать качество сразу на многих моделях и сразу и на обучающих и на валидационных данных в соответствии со всеми метриками
def print_metrics(models, metrics_func, train_data, val_data, metrics_name, models_name):
    results = np.zeros((2 * len(models), len(metrics_func)))
    data_name = []
    for m in models_name:
        data_name.extend([m + " Train", m + " Validation"])
    for m_num, model in enumerate(models):
        for row, sample in enumerate([train_data, val_data]):
            results[row + m_num * 2] = evaluate(model, metrics_func, sample[0], sample[1])
    results = pd.DataFrame(results, columns=metrics_name, index=data_name)
    return results


#Утилита для визуализации того, насколько хорошо мы предсказываем
def draw_predictions(y_true, y_pred, model_name=None):
    if model_name is None:
        model_name = "Model"
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.xlim([5, 50])
    plt.ylim([5, 50])
    sns.regplot(x=y_true, y=y_pred, robust=True,
                label=model_name,
                scatter_kws={"zorder": 10}, line_kws={"zorder": 15})
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predictions")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=5, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.legend()

model_lr_sklearn = LinearRegression()
model_lr_sklearn.fit(X_train_scaled, y_train)

print_metrics(models=[model_lr_sklearn.predict],
              metrics_func=metrics_func,
              train_data=(X_train_tensor, y_train_tensor),
              val_data=(X_val_tensor, y_val_tensor),
              metrics_name=["MSE", "MAPE"],
              models_name=["Sk LR"])

#Нарисуем предсказания
draw_predictions(
    y_true=y_val,
    y_pred=model_lr_sklearn.predict(X_val_scaled),
    model_name="Sklearn LR",
)

model_lr = nn.Sequential(
    nn.Linear(in_features=n_features, out_features=1),
)
#Линейная регрессия - однаиз немногих моделей, оптимальные веса для которой найти можно точно(с помощью конкретнойформулы).
opt_lr = optim.SGD(params=model_lr.parameters(), lr=0.001)
batch_size_lr = 16

#Тренируем модель
epochs_lr = 1000
history_lr_train = []
history_lr_val = []

for epoch in tqdm(range(epochs_lr)):
    for i in range((n_data - 1) // batch_size_lr + 1):
        # формирование батча данных
        start_i = i * batch_size_lr
        end_i = start_i + batch_size_lr
        Xb = X_train_tensor[start_i:end_i]
        yb = y_train_tensor[start_i:end_i]

        # forward pass: делаем предсказания
        pred = model_lr(Xb)
        # forward pass: считаем ошибку
        loss = loss_func(pred, yb)

        # backward pass: считаем градиенты
        loss.backward()

        # обновление весов
        opt_lr.step()
        opt_lr.zero_grad()

    history_lr_train.append(evaluate(model_lr, metrics_func, X_train_tensor, y_train_tensor))
    history_lr_val.append(evaluate(model_lr, metrics_func, X_val_tensor, y_val_tensor))

history_lr_train = np.array(history_lr_train)
history_lr_val = np.array(history_lr_val)

#Нарисуем как менялась функция ошибки по ходу обучения
plt.figure(figsize=(10, 8))
plt.plot(history_lr_train[:, 0], label="LR Train", color="blue")
plt.plot(history_lr_val[:, 0], label="LR Validation", color="orange")
plt.legend(frameon=True)
plt.ylim([0, 75])
plt.ylabel("MSE")
plt.xlabel("Epoch");

print_metrics(models=[model_lr_sklearn.predict, model_lr],
              metrics_func=metrics_func,
              train_data=(X_train_tensor, y_train_tensor),
              val_data=(X_val_tensor, y_val_tensor),
              metrics_name=["MSE", "MAPE"],
              models_name=["Sk LR", "LR"])

Пример предсказания
X_val.tail(1)
y_val.tail(1)

with torch.no_grad():
    print(model_lr(X_val_tensor[-1:]))
tensor([[23.2577]])
X_val_tensor[12]
torch.Size([13])
x_t = torch.Tensor([0.5141, 0.0000, 0.6430, 0.0000, 0.6337, 0.1334, 1.0000, 0.0481, 1.0000,
                    0.9141, 0.8085, 0.2218, 0.9727])
    with torch.no_grad():
        print(model_lr(x_t))
    tensor([-3.0605])

#Нарисуем предсказания
    with torch.no_grad():
        draw_predictions(
            y_true=y_val,
            y_pred=np.array(model_lr(X_val_tensor)).flatten(),
            model_name="PyTorch LR",)

model_mlp_3 = nn.Sequential(
    nn.Linear(in_features=n_features, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=1)
)

opt_mlp_3 = optim.SGD(params=model_mlp_3.parameters(), lr=0.0001)
batch_size_mlp_3 = 16

#Тренируем модель
epochs_mlp_3 = 1000
history_mlp_3_train = []
history_mlp_3_val = []

for epoch in tqdm(range(epochs_mlp_3)):
    for i in range((n_data - 1) // batch_size_mlp_3 + 1):
        start_i = i * batch_size_mlp_3
        end_i = start_i + batch_size_mlp_3
        Xb = X_train_tensor[start_i:end_i]
        yb = y_train_tensor[start_i:end_i]
        pred = model_mlp_3(Xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt_mlp_3.step()
        opt_mlp_3.zero_grad()

    history_mlp_3_train.append(evaluate(model_mlp_3, metrics_func, X_train_tensor, y_train_tensor))
    history_mlp_3_val.append(evaluate(model_mlp_3, metrics_func, X_val_tensor, y_val_tensor))

history_mlp_3_train = np.array(history_mlp_3_train)
history_mlp_3_val = np.array(history_mlp_3_val)

#Нарисуем как менялась функция ошибки по ходу обучения и сравним это с предыдущей моделью(PyTorch Linear Regression)[]
plt.figure(figsize=(10, 8))
plt.plot(history_lr_train[:, 0], label="LR Train", color="blue")
plt.plot(history_lr_val[:, 0], label="LR Validation", color="orange")
plt.plot(history_mlp_3_train[:, 0], label="MLP-3 Train", color="blue", linestyle="--")
plt.plot(history_mlp_3_val[:, 0], label="MLP-3 Validation", color="orange", linestyle="--")
plt.legend(frameon=True)
plt.ylim([0, 75])
plt.ylabel("MSE")
plt.xlabel("Epoch");


print_metrics(models=[model_lr_sklearn.predict, model_lr, model_mlp_3],
              metrics_func=metrics_func,
              train_data=(X_train_tensor, y_train_tensor),
              val_data=(X_val_tensor, y_val_tensor),
              metrics_name=["MSE", "MAPE"],
              models_name=["Sk LR", "LR", "MLP-3"])


with torch.no_grad():
    print(model_mlp_3(X_val_tensor[-1:]))
tensor([[22.0474]])

with torch.no_grad():
    draw_predictions(
        y_true=y_val,
        y_pred=np.array(model_mlp_3(X_val_tensor)).flatten(),
        model_name="PyTorch MLP-3", )

#Сохраним модель для последующего использования
torch.save(model_mlp_3, "mlp3.pth")