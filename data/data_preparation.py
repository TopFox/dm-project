from data.aq_data import aq_data_preparation
from data.meo_data import meo_data_preprocess, generate_nearest_stations

import pandas as pd
import datetime

def data_split() :
    cities = ["bj","ld"]

    for city in cities:
        start_training_date = "2017-1-1 00:00"
        start_testing_date = "2018-3-21 0:00"
        end_date = "2018-3-22 23:00"

        meo = pd.read_csv("./prepared_data/%s_meo_data.csv" %(city))
        aq = pd.read_csv("./prepared_data/%s_aq_data.csv" %(city))		

        meo['date'] = pd.to_datetime(meo['date'])
        aq['time'] = pd.to_datetime(aq['time'])

        meo.set_index("date", inplace=True)
        aq.set_index("time", inplace=True)

        aq_train = aq.loc[start_training_date:end_date] 
        meo_train = meo.loc[start_training_date:end_date]

        aq_test = aq.loc[start_testing_date:end_date]
        meo_test = meo.loc[start_testing_date:end_date]

        aq_train.to_csv("prepared_data/%s_aq_train_data.csv" %(city))
        meo_train.to_csv("prepared_data/%s_meo_train_data.csv" %(city))		


        meo_test.to_csv("prepared_data/%s_meo_test_data.csv" %(city))
        aq_test.to_csv("prepared_data/%s_aq_test_data.csv" %(city))

def prepare_data():
    aq_data_preparation("bj")
    aq_data_preparation("ld")
    meo_data_preprocess("bj")
    meo_data_preprocess("ld")

    data_split()		
    print('Data splitted into training and test sets !')
