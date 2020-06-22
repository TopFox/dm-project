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

def main():

    ld()
    bj()

def ld():

    ld_aq_train_data = pd.read_csv('./prepared_data/ld_aq_train_data.csv')
    ld_meo_train_data = pd.read_csv("./prepared_data/ld_meo_train_data.csv")
    DFgaz = trigaz(ld_aq_train_data.drop('time', axis=1))
    DFmeo = trimeo(ld_meo_train_data.drop('date', axis=1))
    
    daq=[]
    for df in DFgaz:
        
        dfinter=tristation(df)
        for d in dfinter:
            daq.append(d)
    dmeo=[]
    for df in DFmeo:
        
        dfinter=tristation(df)
        for d in dfinter:
            dmeo.append(d)
    pred = []
    for aq in range(0,len(daq)-1):
        
        l=[dmeo[aq][0]]
        l.append(dmeo[aq][0])
        l.append(dmeo[aq+35][0])
        l.append(dmeo[aq+70][0])
        l.append(dmeo[aq+105][0])
        l.append(dmeo[aq+140][0])

      
        df=pd.concat(l,axis=1)
        pred.append(LSTMperso(df))
    df = pd.DataFrame.from_records(pred)
    df.to_csv('./prediction/LSTM_ld_aq.csv')
        

def bj():

    bj_aq_train_data = pd.read_csv('./prepared_data/bj_aq_train_data.csv')
    bj_meo_train_data = pd.read_csv("./prepared_data/bj_meo_train_data.csv")
    #bj_aq_test_data = pd.read_csv('./prepared_data/bj_aq_test_data.csv')
    #bj_meo_test_data = pd.read_csv("./prepared_data/bj_meo_test_data.csv")
    DFgaz = trigaz(bj_aq_train_data.drop('time', axis=1))
    DFmeo = trimeo(bj_meo_train_data.drop('date', axis=1))
    
    daq=[]
    for df in DFgaz:
        
        dfinter=tristation(df)
        for d in dfinter:
            daq.append(d)
    dmeo=[]
    for df in DFmeo:
        
        dfinter=tristation(df)
        for d in dfinter:
            dmeo.append(d)
    pred = []
    for aq in range(0,len(daq)-1):
        
        l=[daq[aq][0]]
        l.append(dmeo[aq][0])
        l.append(dmeo[aq+35][0])
        l.append(dmeo[aq+70][0])
        l.append(dmeo[aq+105][0])
        l.append(dmeo[aq+140][0])
        print(daq[aq][0])
       
       
        df=pd.concat(l,axis=1)
        pred.append(LSTMperso(df))
    
    df = pd.DataFrame.from_records(pred)
    df.to_csv('./prediction/LSTM_bj_aq.csv')
        
        

           




def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def trigaz(data):
    if data.columns[0].split('_')[0] == 'BL0':
        PM2_5 = []
        PM10 = []
        for col in data.columns:
            
            x = col.split('_')
            if x[1]== "PM2.5":
                #print(data[col])
                PM2_5.append(data[col])
            if x[1]=="PM10":
                PM10.append(data[col])
           

        DF_PM2_5 = pd.concat(PM2_5,axis=1)
        DF_PM10 = pd.concat(PM10,axis=1)
     
        return DF_PM2_5, DF_PM10
    else:
        PM2_5 = []
        PM10 = []
        O3 = []
        for col in data.columns:
            
            x = col.split('_')
            if x[1]== "PM2.5":
                #print(data[col])
                PM2_5.append(data[col])
            if x[1]=="PM10":
                PM10.append(data[col])
            if x[1]=="O3":
                O3.append(data[col])

        DF_PM2_5 = pd.concat(PM2_5,axis=1)
        DF_PM10 = pd.concat(PM10,axis=1)
        DF_O3 = pd.concat(O3,axis=1)
        return DF_PM2_5, DF_PM10,DF_O3

def trimeo(data):
    temp = []
    press = []
    humid = []
    w_dir = []
    w_spd = []
    for col in data.columns:
        x = col.split('_')
        if x[1]== "temperature":
            temp.append(data[col])
        if x[1]=="pressure":
            press.append(data[col])
        if x[1]=="humidity":
            humid.append(data[col])
        if x[1]=="wind":
            if x[2]=="direction":
                w_dir.append(data[col])
            else:
                w_spd.append(data[col])

    DF_temp = pd.concat(temp,axis=1)
    DF_press = pd.concat(press,axis=1)
    DF_humid = pd.concat(humid,axis=1)
    DF_w_dir = pd.concat(w_dir,axis=1)
    DF_w_spd = pd.concat(w_spd,axis=1)
    return DF_temp, DF_press,DF_humid,DF_w_dir,DF_w_spd


def tristation(data):
    if data.columns[0].split('_')[0] == 'BL0':
        BL0 = []
        CD1 = []
        CD9 = []
        GN0 = []
        GN3 = []
        GR4 = []
        GR9 = []
        HV1 = []
        KF1 = []
        LW2 = []
        MY7 = []
        ST5 = []
        TH4 = []
        for col in data.columns:
            x=col.split('_')
            if x[0]== 'BL0':
                BL0.append(data[col])
            if x[0]== 'CD1':
                CD1.append(data[col])
            if x[0]== 'CD9':
                CD9.append(data[col])
            if x[0]== 'GN0':
                GN0.append(data[col])
            if x[0]== 'GN3':
                GN3.append(data[col])
            if x[0]== 'GR4':
                GR4.append(data[col])
            if x[0]== 'GR9':
                GR9.append(data[col])
            if x[0]== 'HV1':
                HV1.append(data[col])
            if x[0]== 'KF1':
                KF1.append(data[col])
            if x[0]== 'LW2':
                LW2.append(data[col])
            if x[0]== 'MY7':
                MY7.append(data[col])
            if x[0]== 'ST5':
                ST5.append(data[col])
            if x[0]== 'TH4':
                TH4.append(data[col])
        DF_BL0 = pd.concat(BL0,axis=1)
        DF_CD1 = pd.concat(CD1,axis=1)
        DF_CD9 = pd.concat(CD9,axis=1)
        DF_GN0 = pd.concat(GN0,axis=1)
        DF_GN3 = pd.concat(GN3,axis=1)
        DF_GR4 = pd.concat(GR4,axis=1)
        DF_GR9 = pd.concat(GR9,axis=1)
        DF_HV1 = pd.concat(HV1,axis=1)
        DF_KF1 = pd.concat(KF1,axis=1)
        DF_LW2 = pd.concat(LW2,axis=1)
        DF_MY7 = pd.concat(MY7,axis=1)
        DF_ST5 = pd.concat(ST5,axis=1)
        DF_TH4 = pd.concat(TH4,axis=1)

        return DF_BL0 ,DF_CD1,DF_CD9 ,DF_GN0 ,DF_GN3 ,DF_GR4 , DF_GR9 ,DF_HV1 ,DF_KF1 ,DF_LW2 ,DF_MY7 ,DF_ST5 ,DF_TH4 
    else:
        L1,L2,L3,L4,L5,L6,L7,L8,L9,L10 = [],[],[],[],[],[],[],[],[],[]
        L11,L12,L13,L14,L15,L16,L17,L18,L19,L20 = [],[],[],[],[],[],[],[],[],[]
        L21,L22,L23,L24,L25,L26,L27,L28,L29,L30 = [],[],[],[],[],[],[],[],[],[],
        L31,L32,L33,L34,L35 = [],[],[],[],[],
        for col in data.columns:
            x=col.split('_')
            if x[0]=='fengtaihuayuan':
                L1.append(data[col])
            if x[0]=='yizhuang':
                L2.append(data[col])
            if x[0]=='yongledian':
                L3.append(data[col])
            if x[0]=='miyun':
                L4.append(data[col])
            if x[0]=='nongzhanguan':
                L5.append(data[col])
            if x[0]=='wanliu':
                L6.append(data[col])
            if x[0]=='pinggu':
                L7.append(data[col])
            if x[0]=='badaling':
                L8.append(data[col])
            if x[0]=='mentougou':
                L9.append(data[col])
            if x[0]=='liulihe':
                L10.append(data[col])
            if x[0]=='dingling':
                L11.append(data[col])
            if x[0]=='miyunshuiku':
                L12.append(data[col])
            if x[0]=='nansanhuan':
                L13.append(data[col])
            if x[0]=='wanshouxigong':
                L14.append(data[col])
            if x[0]=='daxing':
                L15.append(data[col])
            if x[0]=='yanqin':
                L16.append(data[col])
            if x[0]=='yongdingmennei':
                L17.append(data[col])
            if x[0]=='qianmen':
                L18.append(data[col])
            if x[0]=='zhiwuyuan':
                L19.append(data[col])
            if x[0]=='dongsi':
                L20.append(data[col])
            if x[0]=='dongsihuan':
                L21.append(data[col])
            if x[0]=='guanyuan':
                L22.append(data[col])
            if x[0]=='donggaocun':
                L23.append(data[col])
            if x[0]=='aotizhongxin':
                L24.append(data[col])
            if x[0]=='fangshan':
                L25.append(data[col])
            if x[0]=='tongzhou':
                L26.append(data[col])
            if x[0]=='xizhimenbei':
                L27.append(data[col])
            if x[0]=='huairou':
                L28.append(data[col])
            if x[0]=='tiantan':
                L29.append(data[col])
            if x[0]=='shunyi':
                L30.append(data[col])
            if x[0]=='yungang':
                L31.append(data[col])
            if x[0]=='beibuxinqu':
                L32.append(data[col])
            if x[0]=='pingchang':
                L33.append(data[col])
            if x[0]=='yufa':
                L34.append(data[col])
            if x[0]=='gucheng':
                L35.append(data[col])
        
        DF_L1 = pd.concat(L1,axis=1)
        DF_L2 = pd.concat(L2,axis=1)
        DF_L3 = pd.concat(L3,axis=1)
        DF_L4 = pd.concat(L4,axis=1)
        DF_L5 = pd.concat(L5,axis=1)
        DF_L6 = pd.concat(L6,axis=1)
        DF_L7 = pd.concat(L7,axis=1)
        DF_L8 = pd.concat(L8,axis=1)
        DF_L9 = pd.concat(L9,axis=1)
        DF_L10 = pd.concat(L10,axis=1)
        DF_L11 = pd.concat(L11,axis=1)
        DF_L12 = pd.concat(L12,axis=1)
        DF_L13 = pd.concat(L13,axis=1)
        DF_L14 = pd.concat(L14,axis=1)
        DF_L15 = pd.concat(L15,axis=1)
        DF_L16 = pd.concat(L16,axis=1)
        DF_L17 = pd.concat(L17,axis=1)
        DF_L18 = pd.concat(L18,axis=1)
        DF_L19 = pd.concat(L19,axis=1)
        DF_L20 = pd.concat(L20,axis=1)
        DF_L21 = pd.concat(L21,axis=1)
        DF_L22 = pd.concat(L22,axis=1)
        DF_L23 = pd.concat(L23,axis=1)
        DF_L24 = pd.concat(L24,axis=1)
        DF_L25 = pd.concat(L25,axis=1)
        DF_L26 = pd.concat(L26,axis=1)
        DF_L27 = pd.concat(L27,axis=1)
        DF_L28 = pd.concat(L28,axis=1)
        DF_L29 = pd.concat(L29,axis=1)
        DF_L30 = pd.concat(L30,axis=1)
        DF_L31 = pd.concat(L31,axis=1)
        DF_L32 = pd.concat(L32,axis=1)
        DF_L33 = pd.concat(L33,axis=1)
        DF_L34 = pd.concat(L34,axis=1)
        DF_L35 = pd.concat(L35,axis=1)
        return L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16,L17,L18,L19,L20,L21,L22,L23,L24,L25,L26,L27,L28,L29,L30,L31,L32,L33,L34,L35 


            

        

            
    
def LSTMperso(dataset):

    
    # manually specify column names
    dataset.columns = ['PM2.5', 'temp', 'press','humitidy', 'wnd_dir', 'wnd_spd']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['PM2.5'].fillna(0, inplace=True)

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
    # split into input and outputs
    # features take all values except the var1
  
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = valuestest[:, :-1], valuestest[:, -1]
    validX, validY = valid[:, :-1], valid[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    validX = validX.reshape((validX.shape[0], 1, validX.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    #print(y_train.shape[0])
    model.add(LSTM(128,input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(1))

    model.compile(loss='mae',optimizer='adam')
    start = time.time()
    history = model.fit(train_X, train_y, epochs=25, batch_size=72, validation_data=(validX, validY), verbose=0, shuffle=False)

    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    #plt.show()
    
    #for i in range(0,47):
        # make a prediction
        
    yhat = model.predict(test_X)
    '''
    print(yhat.shape)
    print(test_X.shape)
    tmp = np.reshape(test_X,[test_X.shape[0],test_X.shape[1],test_X.shape[2]+1])
    tmp[:,:,289] = yhat
    #test_X = np.insert(test_X,-1,yhat.transpose(),axis=0)
    rolleddown = tmp[:,:,1:] + tmp[:,:,:1]
    test_X = np.reshape(rolleddown,[test_X.shape[0],test_X.shape[1],test_X.shape[2]])
    
    print("banane")
    
    print(test_X)
    print(ban)'''

        
        
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    return inv_yhat
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

