from pathlib import Path
import pandas as pd
import numpy as np
import math
import time
import keras
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
import matplotlib.pyplot as plt

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def predict_LSTM():
    cities = ['ld','bj']
    for city in cities:
        aq_train_data = pd.read_csv('./prepared_data/%s_aq_train_data.csv' %(city))
        meo_train_data = pd.read_csv("./prepared_data/%s_meo_train_data.csv" %(city))
        aq_test_data = pd.read_csv("./prepared_data/%s_aq_test_data.csv" %(city))   

        station_names = get_stations_names(aq_train_data.columns)

        pred = []
        for station_name in station_names:
            print(station_name)
            for col in aq_train_data.columns:
                splitted_col = col.split('_')
                if station_name == splitted_col[0]:
                    list_columns = []
                    for meo_col in meo_train_data.columns:
                        splitted_meo_col = meo_col.split('_')
                        if station_name == splitted_meo_col[0]:
                            list_columns.append(meo_train_data[meo_col])
                    list_columns.append(aq_train_data[col])
                    df = pd.concat(list_columns, axis=1)
                    y_pred = LSTMperso(df)
                    y_pred = np.append(col,y_pred)
                    pred.append(y_pred)
                    print(y_pred)
                    print(vagine)
                    
        df_pred_test = pd.DataFrame(pred).T
        time_column = aq_test_data['time']
        time_column.loc[-1] = 'time'
        time_column.index = time_column.index + 1  # shifting index
        time_column.sort_index(inplace=True) 
        df_pred_test.insert(0, 'time', time_column)
        df_pred_test.columns = df_pred_test.iloc[0]
        df_pred_test.set_index('time').to_csv("./prediction/lstm_%s_aq.csv" %(city), header=False)


def get_stations_names(columns):
    station_names_with_duplicates = []
    for col in columns:
        name = col.split('_')
        station_names_with_duplicates.append(name[0])
    station_names = sorted(set(station_names_with_duplicates))
    station_names.remove('time')
    return station_names

def LSTMperso(dataset):
    # manually specify column names
    dataset.columns = ['temp', 'press','humitidy', 'wnd_dir', 'wnd_spd', 'Pollution']
    dataset.index.name = 'date'

    values = dataset.values

    # frame as supervised learning
    reframed = series_to_supervised(values, 48, 1)
    
    # We select only the last 48 hours
    test = reframed.tail(48)
    
    values = reframed.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(values[:,:-1])
    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    valuestest = test.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(valuestest[:,:-1])
    scaled_label = scaler.fit_transform(valuestest[:,-1].reshape(-1,1))
    valuestest = np.column_stack((scaled_features, scaled_label))

    n_train_hours = round(np.size(values,0)*2/3)
    train = values[:n_train_hours, :]
    valid = values[n_train_hours:, :]
  
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = valuestest[:, :-1], valuestest[:, -1]
    validX, validY = valid[:, :-1], valid[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    validX = validX.reshape((validX.shape[0], 1, validX.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(128,input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(1))

    model.compile(loss='mae',optimizer='adam')
    model.fit(train_X, train_y, epochs=150, batch_size=72, validation_data=(validX, validY), verbose=0, shuffle=False)

    yhat = model.predict(test_X)

    # invert scaling for forecast
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    return inv_yhat

