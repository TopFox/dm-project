from pathlib import Path
import pandas as pd
import numpy as np
import math
import time
import keras
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

#matplotlib notebook
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.preprocessing.sequence import TimeseriesGenerator

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        #cols.append(df[data.shape[0]-i:])
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        #cols.append(df[data.shape[0]-i:])
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

def predict_SVR():
    cities = ['ld','bj']
    for city in cities:
        aq_test_data = pd.read_csv("./prepared_data/%s_aq_test_data.csv" %(city))
        aq_train_data = pd.read_csv('./prepared_data/%s_aq_train_data.csv' %(city))
        meo_train_data = pd.read_csv("./prepared_data/%s_meo_train_data.csv" %(city))

        station_names = get_stations_names(aq_train_data.columns)

        pred = []
        for station_name in station_names:
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
                    y_pred = np.flipud(SVRperso(df))
                    y_pred = np.append(col,y_pred)
                    pred.append(y_pred)
                    
        df_pred_test = pd.DataFrame(pred).T
        time_column = aq_test_data['time']
        time_column.loc[-1] = 'time'
        time_column.index = time_column.index + 1  # shifting index
        time_column.sort_index(inplace=True) 
        df_pred_test.insert(0, 'time', time_column)
        df_pred_test.columns = df_pred_test.iloc[0]
        df_pred_test.set_index('time').to_csv("./prediction/svr_%s_aq.csv" %(city), header=False)


def get_stations_names(columns):
    station_names_with_duplicates = []
    for col in columns:
        name = col.split('_')
        station_names_with_duplicates.append(name[0])
    station_names = sorted(set(station_names_with_duplicates))
    station_names.remove('time')
    return station_names
            
def SVRperso(dataset):
    # manually specify column names
    dataset.columns = ['temp', 'press','humitidy', 'wnd_dir', 'wnd_spd', 'Pollution']
    dataset.index.name = 'date'

    values = dataset.values

    # frame as supervised learning
    reframed = series_to_supervised(values, 48, 1)
    
    values = reframed.values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(values[:,:-1])
    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    
    test = values[-48:,:]

    train = values
  
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
   
    x= train_X
    y= train_y


    regr = SVR(C=2,epsilon=0.01,kernel='rbf',gamma=0.1,tol=0.001, verbose=0, shrinking=True, max_iter = 10000)
    regr.fit(x,y)
    data_pred = regr.predict(test_X)
    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1)) 

    return y_pred
        

            
    
