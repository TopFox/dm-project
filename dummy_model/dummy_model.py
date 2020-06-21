import pandas as pd

def train_dummy():
    ld_aq_train_data = pd.read_csv('./prepared_data/ld_aq_train_data.csv')
    bj_aq_train_data = pd.read_csv('./prepared_data/bj_aq_train_data.csv')

    ld_mean = ld_aq_train_data.drop('time', axis=1).mean(axis=0)
    bj_mean = bj_aq_train_data.drop('time', axis=1).mean(axis=0)
    return ld_mean, bj_mean

def predict_dummy():
    ld_aq_prediction = pd.read_csv('./prepared_data/ld_aq_test_data.csv')
    bj_aq_prediction = pd.read_csv('./prepared_data/bj_aq_test_data.csv')

    ld_mean, bj_mean = train_dummy()

    for column_name in ld_aq_prediction.drop('time', axis=1).columns :
        ld_aq_prediction[column_name] = ld_mean.loc[column_name]
    for column_name in bj_aq_prediction.drop('time', axis=1).columns :
        bj_aq_prediction[column_name] = bj_mean.loc[column_name]
    
    ld_aq_prediction.set_index('time').to_csv("./prediction/dummy_ld_aq.csv")
    bj_aq_prediction.set_index('time').to_csv("./prediction/dummy_bj_aq.csv")



