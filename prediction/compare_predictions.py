import pandas as pd
import numpy as np

def  compare_predictions(method):
    ld_aq_test_data = pd.read_csv('./prepared_data/ld_aq_test_data.csv')
    ld_aq_prediction = pd.read_csv('./prediction/%s_ld_aq.csv' %(method))
    bj_aq_test_data = pd.read_csv('./prepared_data/bj_aq_test_data.csv')
    bj_aq_prediction = pd.read_csv('./prediction/%s_bj_aq.csv' %(method))


    ld_mse_city_polluant = np.mean((ld_aq_test_data.drop('time', axis=1) - ld_aq_prediction.drop('time', axis=1))**2)
    bj_mse_city_polluant = np.mean((bj_aq_test_data.drop('time', axis=1) - bj_aq_prediction.drop('time', axis=1))**2)

    ld_PM25_array = []
    ld_PM10_array = []
    print(len(ld_mse_city_polluant))
    for i in range(0,len(ld_mse_city_polluant)-1):
        if 'PM2.5' in ld_mse_city_polluant.index[i]:
            ld_PM25_array.append(ld_mse_city_polluant[i])
        if 'PM10' in ld_mse_city_polluant.index[i]:
            ld_PM10_array.append(ld_mse_city_polluant[i])

    bj_PM25_array = []
    bj_PM10_array = []
    bj_O3_array = []
    for i in range(0,len(bj_mse_city_polluant)-1):
        #print(i)
        if 'PM2.5' in bj_mse_city_polluant.index[i]:
            bj_PM25_array.append(bj_mse_city_polluant[i])
        if 'PM10' in bj_mse_city_polluant.index[i]:
            bj_PM10_array.append(bj_mse_city_polluant[i])
        if 'O3' in bj_mse_city_polluant.index[i]:
            bj_O3_array.append(bj_mse_city_polluant[i])
    
    print(bj_O3_array)
    
    print("%s method's predictions results :" %(method))
    print('PM 2.5 RMSE in London :', np.mean(ld_PM25_array))
    print('PM 10 RMSE in London :', np.mean(ld_PM10_array))
    print('PM 2.5 RMSE in Beijing :', np.mean(bj_PM25_array))
    print('PM 10 RMSE in Beijing :', np.mean(bj_PM10_array))
    print('O3 RMSE in Beijing :', np.mean(bj_O3_array))

    # print(ld_mse_city_polluant.index)
    # for 
    # ld_polluant_test =
    # ld_polluant_prediction = 
    # ld_mse_polluant = np.mean((bj_aq_test_data.drop('time', axis=1) - bj_aq_prediction.drop('time', axis=1))**2)

    # print('RMSE for London using %s prediction :')
    # print(np.sqrt(ld_mse_city_polluant))
    # print('RMSE for Beijing using %s prediction :')
    # print(np.sqrt(bj_mse_city_polluant))