import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def  compare_predictions(method):
    ld_aq_test_data = pd.read_csv('./prepared_data/ld_aq_test_data.csv').sort_index(axis=1).drop('time', axis=1)
    ld_aq_prediction = pd.read_csv('./prediction/%s_ld_aq.csv' %(method)).sort_index(axis=1).drop('time', axis=1)
    bj_aq_test_data = pd.read_csv('./prepared_data/bj_aq_test_data.csv').sort_index(axis=1).drop('time', axis=1)
    bj_aq_prediction = pd.read_csv('./prediction/%s_bj_aq.csv' %(method)).sort_index(axis=1).drop('time', axis=1)

    def smape(A, F):
        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

    ld_rmse_city_polluant = np.sqrt(np.mean((ld_aq_test_data - ld_aq_prediction)**2))
    bj_rmse_city_polluant = np.sqrt(np.mean((bj_aq_test_data - bj_aq_prediction)**2))

    ld_smape_city_polluant = smape(ld_aq_test_data, ld_aq_prediction)
    bj_smape_city_polluant = smape(bj_aq_test_data, bj_aq_prediction)

    ld_PM25_rmse_array = []
    ld_PM10_rmse_array = []
    ld_PM25_smape_array = []
    ld_PM10_smape_array = []
    for i in range(0,len(ld_rmse_city_polluant)-1):
        if 'PM2.5' in ld_rmse_city_polluant.index[i]:
            ld_PM25_rmse_array.append(ld_rmse_city_polluant[i])
            ld_PM25_smape_array.append(ld_smape_city_polluant[i])
        if 'PM10' in ld_rmse_city_polluant.index[i]:
            ld_PM10_rmse_array.append(ld_rmse_city_polluant[i])
            ld_PM10_smape_array.append(ld_smape_city_polluant[i])

    bj_PM25_rmse_array = []
    bj_PM10_rmse_array = []
    bj_O3_rmse_array = []
    bj_PM25_smape_array = []
    bj_PM10_smape_array = []
    bj_O3_smape_array = []
    for i in range(0,len(bj_rmse_city_polluant)-1):
        if 'PM2.5' in bj_rmse_city_polluant.index[i]:
            bj_PM25_rmse_array.append(bj_rmse_city_polluant[i])
            bj_PM25_smape_array.append(bj_smape_city_polluant[i])
        if 'PM10' in bj_rmse_city_polluant.index[i]:
            bj_PM10_rmse_array.append(bj_rmse_city_polluant[i])
            bj_PM10_smape_array.append(bj_smape_city_polluant[i])
        if 'O3' in bj_rmse_city_polluant.index[i]:
            bj_O3_rmse_array.append(bj_rmse_city_polluant[i])
            bj_O3_smape_array.append(bj_smape_city_polluant[i])
    
    print("Predictions of %s method results :" %(method))
    print('PM 2.5 RMSE in London :', np.mean(ld_PM25_rmse_array))
    print('PM 10 RMSE in London :', np.mean(ld_PM10_rmse_array))
    print('PM 2.5 RMSE in Beijing :', np.mean(bj_PM25_rmse_array))
    print('PM 10 RMSE in Beijing :', np.mean(bj_PM10_rmse_array))
    print('O3 RMSE in Beijing :', np.mean(bj_O3_rmse_array))
    print('')
    print('PM 2.5 SMAPE in London :', np.mean(ld_PM25_smape_array))
    print('PM 10 SMAPE in London :', np.mean(ld_PM10_smape_array))
    print('PM 2.5 SMAPE in Beijing :', np.mean(bj_PM25_smape_array))
    print('PM 10 SMAPE in Beijing :', np.mean(bj_PM10_smape_array))
    print('O3 SMAPE in Beijing :', np.mean(bj_O3_smape_array))
    print('-------------------------------------------------------')

    fig, axs = plt.subplots(2,3,figsize=(16,10))
    ld_aq_test_data['BL0_PM2.5'].plot(ax=axs[1,0], label='Actual pollution')
    ld_aq_prediction['BL0_PM2.5'].plot(ax=axs[1,0], label='Prediction')
    axs[1,0].set_title('PM 2.5 variations in the station BL0')
    axs[1,0].set_xlabel('Time (in hours)')
    axs[1,0].set_ylabel('PM 2.5 (in microgram/m^3')
    axs[1,0].legend(loc='best')


    bj_aq_test_data['pinggu_PM2.5'].plot(ax=axs[0,0], label='Actual pollution')
    bj_aq_prediction['pinggu_PM2.5'].plot(ax=axs[0,0], label='Prediction')
    axs[0,0].set_title('PM 2.5 variations in the station pinggu')
    axs[0,0].set_xlabel('Time (in hours)')
    axs[0,0].set_ylabel('PM 2.5 (in microgram/m^3')
    axs[0,0].legend(loc='best')

    ld_aq_test_data['BL0_PM10'].plot(ax=axs[1,1], label='Actual pollution')
    ld_aq_prediction['BL0_PM10'].plot(ax=axs[1,1], label='Prediction')
    axs[1,1].set_title('PM 10 variations in the station BL0')
    axs[1,1].set_xlabel('Time (in hours)')
    axs[1,1].set_ylabel('PM 10 (in microgram/m^3')
    axs[1,1].legend(loc='best')

    bj_aq_test_data['pinggu_PM10'].plot(ax=axs[0,1], label='Actual pollution')
    bj_aq_prediction['pinggu_PM10'].plot(ax=axs[0,1], label='Prediction')
    axs[0,1].set_title('PM 10 variations in the station pinggu')
    axs[0,1].set_xlabel('Time (in hours)')
    axs[0,1].set_ylabel('PM 10 (in microgram/m^3')
    axs[0,1].legend(loc='best')


    bj_aq_test_data['pinggu_O3'].plot(ax=axs[0,2], label='Actual pollution')
    bj_aq_prediction['pinggu_O3'].plot(ax=axs[0,2], label='Prediction')
    axs[0,2].set_title('O3 variations in the station pinggu')
    axs[0,2].set_xlabel('Time (in hours)')
    axs[0,2].set_ylabel('O3 (in microgram/m^3')
    axs[0,2].legend(loc='best')

    fig.suptitle('Predictions of %s model in two stations' %(method))
    fig.savefig('%s_prediction.png' %(method))

    #return ld_rmse_city_polluant, bj_rmse_city_polluant