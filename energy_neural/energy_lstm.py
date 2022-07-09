# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
#
# import plotly.graph_objs as go
# from plotly.offline import iplot

from energy_neural.nns import RNNModel, LSTMModel, GRUModel, SequenceDataset, ShallowLSTMModel

from energy_neural.nns import Optimization
from sklearn.linear_model import LinearRegression

torch.manual_seed(0)
np.random.seed(0)


# def plot_dataset(df, title):
#     data = []
#
#     value = go.Scatter(
#         x=df.index,
#         y=df.value,
#         mode="lines",
#         name="values",
#         marker=dict(),
#         text=df.index,
#         line=dict(color="rgba(0,0,0, 0.3)"),
#     )
#     data.append(value)
#
#     layout = dict(
#         title=title,
#         xaxis=dict(title="Date", ticklen=5, zeroline=False),
#         yaxis=dict(title="Value", ticklen=5, zeroline=False),
#     )
#
#     fig = dict(data=data, layout=layout)
#     iplot(fig)


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"{device}" " is available.")
# %%
df = pd.read_csv('../data/PJME_hourly.csv')

df = df.set_index(['Datetime'])
df = df.rename(columns={'PJME_MW': 'value'})

df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()

# plot_dataset(df, title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')

print(df)

def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n

input_dim = 100

df_timelags = generate_time_lags(df, input_dim)
df_timelags


df_features = (
                df
                .assign(hour = df.index.hour)
                .assign(day = df.index.day)
                .assign(month = df.index.month)
                .assign(day_of_week = df.index.dayofweek)
                .assign(week_of_year = df.index.week)
              )

df_features


def onehot_encode_pd(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col)

    return pd.concat([df, dummies], axis=1).drop(columns=cols)


df_features = onehot_encode_pd(df_features, ['week_of_year'])

df_features

'''
## TIME FEATURES ##
Though quite useful to encode categorical features, one-hot encoding does not fully capture the cyclical patterns in DateTime features. 
It simply creates categorical buckets, if you will, and lets the model learn from these seemingly independent features. 
Encoding the day of the week in a similar manner, for instance, loses the information that Monday is closer to Tuesday than Wednesday.
For some use cases, this may not matter too much, indeed. In fact, with enough data, training time, and model complexity, the model may learn such relationships between such features independently. But there is also another way.

The problem simply becomes how can we tell algorithms that the hours 23 and 0 are as close as hour 1 is to hour 2?

'''

def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
             }
    return df.assign(**kwargs).drop(columns=[col_name])

df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
df_features = generate_cyclical_features(df_features, 'day', 31, 1)
df_features = generate_cyclical_features(df_features, 'month', 12, 1)
df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 1)
# df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

# %%
from datetime import date
import holidays

us_holidays = holidays.US()

def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))


df_features = add_holiday_col(df_features, us_holidays)

from sklearn.model_selection import train_test_split
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    '''
    IMPORTANT: Shuffle = FALSE because time dependent
    '''
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col) ## split target from the rest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

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
input_dim = len(X_train.columns)
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

#
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


model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout,
                'device' : device
                }

model = get_model('gru', model_params)
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
opt.train(train_loader, val_loader, batch_size, n_epochs, n_features=input_dim)

# opt.plot_losses()

predictions, values = opt.evaluate(
    test_loader_one,
    batch_size=1,
    n_features=input_dim
)

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    '''
    Need to unscale predictions to be usable

    '''
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


df_result = format_predictions(predictions, values, X_test, scaler)
df_result

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(df):
    result_metrics = {'mae': mean_absolute_error(df.value, df.prediction),
                      'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
                      'r2': r2_score(df.value, df.prediction)}

    print("Mean Absolute Error:       ", result_metrics["mae"])
    print("Root Mean Squared Error:   ", result_metrics["rmse"])
    print("R^2 Score:                 ", result_metrics["r2"])
    return result_metrics


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


'''
### Formatting the predictions
As you may recall, we trained our network with standardized inputs; therefore, all the model's predictions are also scaled. 
Also, after using batching in our evaluation method, all of our predictions are now in batches. 
To calculate error metrics and plot these predictions, 
we need first to reduce these multi-dimensional tensors to a one-dimensional vector, i.e., flatten, 
and then apply inverse_transform() to get the predictions' real values.
'''


