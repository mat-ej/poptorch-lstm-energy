# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from poptorch_energy.nns import RNNModel, LSTMModel, GRUModel, SequenceDataset, ShallowLSTMModel

from poptorch_energy.nns import Optimization
from sklearn.linear_model import LinearRegression
from poptorch_energy.paths import *
import poptorch
import holidays

from poptorch_energy.util import generate_time_lags, onehot_encode_pd, generate_cyclical_features, train_val_test_split, \
    add_holiday_col, format_predictions, calculate_metrics, feature_label_split

torch.manual_seed(0)
np.random.seed(0)

# %%

file_to_open = data_path / "PJME_hourly.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"{device}" " is available.")
# %%
df = pd.read_csv(file_to_open)

df = df.set_index(['Datetime'])
df = df.rename(columns={'PJME_MW': 'value'})

df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()

# plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')

# print(df)

time_lags = 100
df_timelags = generate_time_lags(df, time_lags)

df_features = (
                df
                .assign(hour = df.index.hour)
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
              )

df_features = onehot_encode_pd(df_features, ['week_of_year'])

'''
## TIME FEATURES ##
Though quite useful to encode categorical features, one-hot encoding does not fully capture the cyclical patterns in DateTime features. 
It simply creates categorical buckets, if you will, and lets the model learn from these seemingly independent features. 
Encoding the day of the week in a similar manner, for instance, loses the information that Monday is closer to Tuesday than Wednesday.
For some use cases, this may not matter too much, indeed. In fact, with enough data, training time, and model complexity, the model may learn such relationships between such features independently. But there is also another way.

The problem simply becomes how can we tell algorithms that the hours 23 and 0 are as close as hour 1 is to hour 2?

'''

df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
df_features = generate_cyclical_features(df_features, 'day', 31, 1)
df_features = generate_cyclical_features(df_features, 'month', 12, 1)
df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 1)
# df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

# %%

us_holidays = holidays.US()
df_features = add_holiday_col(df_features, us_holidays)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, target_col='value', test_ratio=0.2)

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

scaler = get_scaler('minmax')
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)

# %%
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim

batch_size = 64
time_lags = len(X_train.columns)
output_dim = 1
hidden_dim = 64
layer_dim = 3
# layer_dim = 3
dropout = 0.2
n_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-6
sequence_length = 10

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)


train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

# train = SequenceDataset(train_features, train_targets, sequence_length)
# val = SequenceDataset(val_features, val_targets, sequence_length)
# test = SequenceDataset(test_features, test_targets, sequence_length)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
        "shallowlstm": ShallowLSTMModel,
    }
    return models.get(model.lower())(**model_params)


model_params = {'input_dim': time_lags,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout,
                'device' : device
                }

model = get_model('gru', model_params)

model = poptorch.trainingModel(model)
print(model)
print("training")
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
opt.train(train_loader, val_loader, batch_size, n_epochs, n_features=time_lags)

opt.plot_losses()

predictions, values = opt.evaluate(
    test_loader_one,
    batch_size=1,
    n_features=time_lags
)


df_result = format_predictions(predictions, values, X_test, scaler)
result_metrics = calculate_metrics(df_result)


def build_baseline_model(df, test_ratio, target_col):
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result


print("linreg baseline")
df_baseline = build_baseline_model(df_features, 0.2, 'value')
baseline_metrics = calculate_metrics(df_baseline)

