from pathlib import Path
import pandas as pd
import numpy as np
import math
import time
import keras
import pandas_profiling
from datetime import datetime
from sklearn.model_selection import train_test_split

#matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json

def temp():

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



    bj_aq_train_data = pd.read_csv('./prepared_data/bj_aq_train_data.csv')
    bj_meo_train_data = pd.read_csv("./prepared_data/bj_meo_train_data.csv")

    bj_aq_test_data = pd.read_csv("./prepared_data/bj_aq_test_data.csv")
    bj_meo_test_data = pd.read_csv("./prepared_data/bj_meo_test_data.csv")

    print(bj_meo_train_data.columns)

    a = bj_aq_test_data[["tongzhou_aq_PM2.5"]]
    s = bj_meo_test_data[["tongzhou_aq_temperature","tongzhou_aq_pressure","tongzhou_aq_humidity","tongzhou_aq_wind_direction","tongzhou_aq_wind_speed"]]

    for col in bj_aq_train_data.drop('time', axis=1).columns:

        t=bj_aq_train_data[[col]]
        temp2(dataset)

    t = bj_aq_train_data[["tongzhou_aq_PM2.5"]]
    b = bj_meo_train_data[["tongzhou_aq_temperature","tongzhou_aq_pressure","tongzhou_aq_humidity","tongzhou_aq_wind_direction","tongzhou_aq_wind_speed"]]
    print(t.head(5))
    print(b.head(5))
    dataset= pd.concat((t,b),axis=1)
    

def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

    
def temp2(dataset):

    print(dataset.columns)
    # manually specify column names
    dataset.columns = ['PM2.5', 'temp', 'press','humitidy', 'wnd_dir', 'wnd_spd']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['PM2.5'].fillna(0, inplace=True)


    # summarize first 5 rows
    print('bonjour')
    print(dataset.head(5))
    # save to file
    dataset.to_csv('PM2.5.csv')
    row=0

    dataset = pd.read_csv('PM2.5.csv', header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5]
    i = 1
    # plot each column
    plt.figure()
    for group in groups:
    	plt.subplot(len(groups), 1, i)
    	plt.plot(values[:, group])
    	plt.title(dataset.columns[group], y=0.5, loc='right')
    	i += 1
    plt.show()

    # load dataset
    dataset = pd.read_csv('PM2.5.csv', header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    #encoder = LabelEncoder()
    #values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features

    # frame as supervised learning
    reframed = series_to_supervised(values, 48, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[7,8,9,10,11]], axis=1, inplace=True)
    print(reframed.head(),"aurevoir")

    values = reframed.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(values[:,:-1])
    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    n_train_hours = round(np.size(values,0)*2/3)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    # features take all values except the var1
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


    model = Sequential()
    #print(y_train.shape[0])
    model.add(LSTM(128,input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(1))

    model.compile(loss='mae',optimizer='adam')
    start = time.time()
    history = model.fit(train_X, train_y, epochs=25, batch_size=72, validation_data=(test_X, test_y), verbose=1, shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    end = time.time()
    print('This took {} seconds.'.format(end - start))
    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

