from pathlib import Path
import pandas as pd
import numpy as np
import math
import time
import keras
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import ensemble, datasets
from sklearn.metrics import mean_squared_error

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

def predict_GBRT():
    cities = ['ld','bj']
    best_n_estimators, best_max_depth, best_min_samples_split, best_learning_rate = find_best_parameters()
    for city in cities:
        aq_train_data = pd.read_csv('./prepared_data/%s_aq_train_data.csv' %(city))
        meo_train_data = pd.read_csv("./prepared_data/%s_meo_train_data.csv" %(city))
        aq_test_data = pd.read_csv("./prepared_data/%s_aq_test_data.csv" %(city))   

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
                    y_pred = GBRTperso(df, best_n_estimators, best_max_depth, best_min_samples_split, best_learning_rate, 'ls', True)
                    y_pred = np.append(col,y_pred)
                    pred.append(y_pred)
                                      
        df_pred_test = pd.DataFrame(pred).T
        time_column = aq_test_data['time']
        time_column.loc[-1] = 'time'
        time_column.index = time_column.index + 1  # shifting index
        time_column.sort_index(inplace=True) 
        df_pred_test.insert(0, 'time', time_column)
        df_pred_test.columns = df_pred_test.iloc[0]
        df_pred_test.set_index('time').to_csv("./prediction/gbrt_%s_aq.csv" %(city), header=False)

def get_stations_names(columns):
    station_names_with_duplicates = []
    for col in columns:
        name = col.split('_')
        station_names_with_duplicates.append(name[0])
    station_names = sorted(set(station_names_with_duplicates))
    station_names.remove('time')
    return station_names

def find_best_parameters():
    n_estimators_list = [50, 100, 250]
    max_depth_list = [10, 13, 16]
    learning_rate_list = [0.5, 0.1, 0.01]
    min_samples_split_list = [2, 3, 4]

    best_mean_mse = 10000000

    dataframes = []
    cities = ['ld','bj']

    for city in cities:
        aq_train_data = pd.read_csv('./prepared_data/%s_aq_train_data.csv' %(city))
        meo_train_data = pd.read_csv("./prepared_data/%s_meo_train_data.csv" %(city))

        station_names = get_stations_names(aq_train_data.columns)

        station_name = station_names[0]
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
                dataframes.append(df)

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                for min_samples_split in min_samples_split_list:
                    mse_list = []
                    start = time.time()
                    for df in dataframes:
                        mse_list.append(GBRTperso(df, n_estimators, max_depth, min_samples_split, learning_rate, 'ls', False))
                    mean_mse = np.mean(mse_list)
                    if mean_mse < best_mean_mse:
                        best_mean_mse = mean_mse
                        best_n_estimators = n_estimators
                        best_max_depth = max_depth
                        best_min_samples_split = min_samples_split
                        best_learning_rate = learning_rate

    print('Cross validation over')
    return best_n_estimators, best_max_depth, best_min_samples_split, best_learning_rate

def GBRTperso(dataset, n_estimators, max_depth, min_samples_split, learning_rate, loss, isFinal):
    dataset.columns = ['temp', 'press','humitidy', 'wnd_dir', 'wnd_spd', 'Pollution']
    dataset.index.name = 'date'

    if isFinal:
        values = dataset.values
    else:
        values = dataset.values[-300:,:]

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
    
    params = {'n_estimators' : n_estimators,'max_depth': max_depth, 'min_samples_split': min_samples_split, 'learning_rate': learning_rate,'loss':'ls'}
   
    x= train_X
    y= train_y

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(x,y)
    data_pred = reg.predict(test_X) 
    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1)) 

    if isFinal:
        return y_pred
    
    else :
        y_real = scaler.inverse_transform(test_y.reshape(-1,1))
        mse = mean_squared_error(y_pred,y_real)
        return mse