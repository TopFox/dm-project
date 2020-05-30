import os
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

bj_near_stations = {'aotizhongxin_aq': 'beijing_grid_304',
                     'badaling_aq': 'beijing_grid_224',
                     'beibuxinqu_aq': 'beijing_grid_263',
                     'daxing_aq': 'beijing_grid_301',
                     'dingling_aq': 'beijing_grid_265',
                     'donggaocun_aq': 'beijing_grid_452',
                     'dongsi_aq': 'beijing_grid_303',
                     'dongsihuan_aq': 'beijing_grid_324',
                     'fangshan_aq': 'beijing_grid_238',
                     'fengtaihuayuan_aq': 'beijing_grid_282',
                     'guanyuan_aq': 'beijing_grid_282',
                     'gucheng_aq': 'beijing_grid_261',
                     'huairou_aq': 'beijing_grid_349',
                     'liulihe_aq': 'beijing_grid_216',
                     'mentougou_aq': 'beijing_grid_240',
                     'miyun_aq': 'beijing_grid_392',
                     'miyunshuiku_aq': 'beijing_grid_414',
                     'nansanhuan_aq': 'beijing_grid_303',
                     'nongzhanguan_aq': 'beijing_grid_324',
                     'pingchang_aq': 'beijing_grid_264',
                     'pinggu_aq': 'beijing_grid_452',
                     'qianmen_aq': 'beijing_grid_303',
                     'shunyi_aq': 'beijing_grid_368',
                     'tiantan_aq': 'beijing_grid_303',
                     'tongzhou_aq': 'beijing_grid_366',
                     'wanliu_aq': 'beijing_grid_283',
                     'wanshouxigong_aq': 'beijing_grid_303',
                     'xizhimenbei_aq': 'beijing_grid_283',
                     'yanqin_aq': 'beijing_grid_225',
                     'yizhuang_aq': 'beijing_grid_323',
                     'yongdingmennei_aq': 'beijing_grid_303',
                     'yongledian_aq': 'beijing_grid_385',
                     'yufa_aq': 'beijing_grid_278',
                     'yungang_aq': 'beijing_grid_239',
                     'zhiwuyuan_aq': 'beijing_grid_262'}

ld_near_stations = {'BL0': 'london_grid_409',
                     'BX1': 'london_grid_472',
                     'BX9': 'london_grid_472',
                     'CD1': 'london_grid_388',
                     'CD9': 'london_grid_409',
                     'CR8': 'london_grid_408',
                     'CT2': 'london_grid_409',
                     'CT3': 'london_grid_409',
                     'GB0': 'london_grid_451',
                     'GN0': 'london_grid_451',
                     'GN3': 'london_grid_451',
                     'GR4': 'london_grid_451',
                     'GR9': 'london_grid_430',
                     'HR1': 'london_grid_368',
                     'HV1': 'london_grid_472',
                     'KC1': 'london_grid_388',
                     'KF1': 'london_grid_388',
                     'LH0': 'london_grid_346',
                     'LW2': 'london_grid_430',
                     'MY7': 'london_grid_388',
                     'RB7': 'london_grid_452',
                     'ST5': 'london_grid_408',
                     'TD5': 'london_grid_366',
                     'TH4': 'london_grid_430'}

def load_bj_pred_grid_meo_data(useful_stations):
    '''
    csv_list : a list of strings, string of csv path
    useful_stations : dict of {aq_station : meo_station}
    '''
    path_to_bj_meo = "./data/"
    bj_csv_list  = os.listdir(path_to_bj_meo)

    bj_meo_datas = []

    for csv in bj_csv_list :
        if csv != '.DS_Store' and not csv.startswith("._"):
            path_to_file = path_to_bj_meo + csv
            # print(path_to_file)
            bj_meo_data = pd.read_csv(path_to_file)
            # print(bj_meo_data.columns)

            # 去掉多余信息
            if "longitude" in bj_meo_data.columns :
                bj_meo_data.drop("longitude", axis=1, inplace=True)
            if "latitude" in bj_meo_data.columns :    
                bj_meo_data.drop("latitude", axis=1, inplace=True)
            if "id" in bj_meo_data.columns :
                bj_meo_data.drop("id", axis=1, inplace=True)
            if "weather" in bj_meo_data.columns :
                bj_meo_data.drop("weather", axis=1, inplace=True)
            
            name_pairs = {}
            if "station_id" in bj_meo_data.columns :
                name_pairs["station_id"] = "stationName"
            if "forecast_time" in bj_meo_data.columns :
                name_pairs["forecast_time"] = "utc_time"
            if "wind_speed/kph" in bj_meo_data.columns :
                name_pairs["wind_speed/kph"] = "wind_speed"
            
            bj_meo_data.rename(index=str, columns=name_pairs, inplace=True)
            bj_meo_datas.append(bj_meo_data)

    meo_dataset = pd.concat(bj_meo_datas, ignore_index=True)
    meo_dataset.sort_index(inplace=True)
    meo_dataset.drop_duplicates(subset=None, keep='first', inplace=True)

    bj_grid_meo_dataset, stations, bj_meo_stations = load_grid_meo_data(meo_dataset, useful_stations)

    return bj_grid_meo_dataset, stations, bj_meo_stations

def load_ld_pred_grid_meo_data(useful_stations):
    '''
    csv_list : a list of strings, string of csv path
    useful_stations : dict of {aq_station : meo_station}
    '''
    path_to_ld_meo = "./data/"
    ld_csv_list  = os.listdir(path_to_ld_meo)

    ld_meo_datas = []

    for csv in ld_csv_list :
        if csv != '.DS_Store' and not csv.startswith("._"):
            path_to_file = path_to_ld_meo + csv
            # print(path_to_file)
            ld_meo_data = pd.read_csv(path_to_file)
            # print(ld_meo_data.columns)

            # 去掉多余信息
            if "longitude" in ld_meo_data.columns :
                ld_meo_data.drop("longitude", axis=1, inplace=True)
            if "latitude" in ld_meo_data.columns :    
                ld_meo_data.drop("latitude", axis=1, inplace=True)
            if "id" in ld_meo_data.columns :
                ld_meo_data.drop("id", axis=1, inplace=True)
            if "weather" in ld_meo_data.columns :
                ld_meo_data.drop("weather", axis=1, inplace=True)
            
            name_pairs = {}
            if "station_id" in ld_meo_data.columns :
                name_pairs["station_id"] = "stationName"
            if "forecast_time" in ld_meo_data.columns :
                name_pairs["forecast_time"] = "utc_time"
            if "wind_speed/kph" in ld_meo_data.columns :
                name_pairs["wind_speed/kph"] = "wind_speed"
             
            ld_meo_data.rename(index=str, columns=name_pairs, inplace=True)
            # print(ld_meo_data.columns)
            ld_meo_datas.append(ld_meo_data)

    meo_dataset = pd.concat(ld_meo_datas, ignore_index=True)
    meo_dataset.sort_index(inplace=True)
    meo_dataset.drop_duplicates(subset=None, keep='first', inplace=True)

    ld_grid_meo_dataset, stations, ld_meo_stations = load_grid_meo_data(meo_dataset, useful_stations)

    return ld_grid_meo_dataset, stations, ld_meo_stations

def load_bj_grid_meo_data(useful_stations):
    meo_dataset = pd.read_csv("./data/Beijing_historical_meo_grid.csv")
    meo_dataset.drop("longitude", axis=1, inplace=True)   
    meo_dataset.drop("latitude", axis=1, inplace=True)

    meo_dataset.rename(index=str, columns={"wind_speed/kph": "wind_speed"}, inplace=True)

    meo_dataset.sort_index(inplace=True)
    meo_dataset.drop_duplicates(subset=None, inplace=True)

    bj_grid_meo_dataset, stations, bj_meo_stations = load_grid_meo_data(meo_dataset, useful_stations)

    return bj_grid_meo_dataset, stations, bj_meo_stations

def load_ld_grid_meo_data(useful_stations):

    meo_dataset = pd.read_csv("./data/London_historical_meo_grid.csv")
    meo_dataset.drop("longitude", axis=1, inplace=True)    
    meo_dataset.drop("latitude", axis=1, inplace=True)

    meo_dataset.rename(index=str, columns={"wind_speed/kph" : "wind_speed"}, inplace=True)

    meo_dataset.sort_index(inplace=True)
    meo_dataset.drop_duplicates(subset=None, inplace=True)

    ld_grid_meo_dataset, stations, ld_meo_stations = load_grid_meo_data(meo_dataset, useful_stations)

    return ld_grid_meo_dataset, stations, ld_meo_stations

def load_grid_meo_data(meo_dataset, useful_stations):
    
    meo_dataset["time"] = pd.to_datetime(meo_dataset['utc_time'])
    meo_dataset.set_index("time", inplace=True)
    meo_dataset.drop("utc_time", axis=1, inplace=True)

    # names of all stations
    stations = set(meo_dataset['stationName'])

    # a dict of station aq, Beijing
    meo_stations = {}

    for aq_station_name, meo_station_name in useful_stations.items() :
        if meo_station_name in stations :
            meo_station = meo_dataset[meo_dataset["stationName"]==meo_station_name].copy()
            meo_station.drop("stationName", axis=1, inplace=True)
            if "None" in meo_station.columns :
                meo_station.drop("None", axis=1, inplace=True)

            original_names = meo_station.columns.values.tolist()
            names_dict = {original_name : aq_station_name+"_"+original_name for original_name in original_names}
            meo_station_renamed = meo_station.rename(index=str, columns=names_dict)

            meo_stations[aq_station_name] = meo_station_renamed        

    return meo_dataset, stations, meo_stations

def get_station_locations(stations_df):
    '''
    Get all the locations of stations in stations_df.
    Agrs : 
        stations_df : a dataframe of all station data.
    Return : 
        A list of (station_name, (longitude, latitude))
    '''
    
    locations = []
    station_names = []
    
    if 'station_id' in stations_df.columns:
        station_column_name = 'station_id'
    elif 'stationName' in stations_df.columns:
        station_column_name = 'stationName'
    else :
        print("Can not find station name!")
    
    for j in stations_df.index:
        station_name = stations_df[station_column_name][j]
        if station_name not in station_names:
            station_names.append(station_name)
            longitude = stations_df['longitude'][j]
            latitude = stations_df['latitude'][j]
            location = (longitude, latitude)
            # station_name = stations_df[station_column_name][j]
            locations.append((station_name, location))
    
    return locations

def get_location_lists(locations):
    '''
    Get location list from locations.
    Args : 
        A list with element shape (station_name, (longitude, latitude)).
    Return : 
        Two lists of longitudes and latitudes.
    '''
    longitudes = []
    latitudes = []
    
    for i in range(len(locations)):
        _, (longitude, latitude) = locations[i]
        longitudes.append(longitude)
        latitudes.append(latitude)
        
    return longitudes, latitudes

def find_nearst_meo_station_name(aq_location, meo_locations):
    '''
    From meo stations ans grid meos stations, find the nearest meo station of aq station.
    Args :
        aq_location : an aq station information of (station_name, (longitude, latitude))
        meo_locations : meo information, list of ((station_name, (longitude, latitude)))
    '''
    nearest_station_name = ""
    nearest_distance = 1e10
    
    aq_station_longitude = aq_location[1][0]
    aq_station_latitude = aq_location[1][1]
    
    for station_name, (longitude, latitude) in meo_locations:
        dis = np.sqrt((longitude-aq_station_longitude)**2 + (latitude-aq_station_latitude)**2)
        if dis < nearest_distance:
            nearest_distance = dis
            nearest_station_name = station_name
    
    return nearest_station_name

def get_related_meo_dfs(aq_station_nearest_meo_station, bj_meo_all, bj_grid_meo_all):
    '''
    Get a dict with aq_station_name as key and nearest meo_station meo data as value.
    Args :
        aq_station_nearest_meo_station = {aq_station_name : meo_station_name}
        bj_meo_all is Beijing meo dataframe.
        grid_bj_meo_all is Beijing grid meo dataframe.
    Returns:
        related_meo_dfs = {aq_station_name : meo_station_data_df}
    '''
    related_meo_dfs = {}
    
    bj_meo_all_names = set(bj_meo_all["station_id"].values)
    grid_bj_meo_all_names = set(bj_grid_meo_all["stationName"].values)

    for aq_station, meo_station in aq_station_nearest_meo_station.items():
        if meo_station in bj_meo_all_names:
            related_meo_dfs[aq_station] = bj_meo_all[bj_meo_all['station_id'] == meo_station]
        elif meo_station in grid_bj_meo_all_names:
            related_meo_dfs[aq_station] = bj_grid_meo_all[bj_grid_meo_all['stationName'] == meo_station]
        else :
            print("meo station name not found.")
    
    return related_meo_dfs

def meo_data_preprocess(city="bj"): 
    if city == "bj" :
        grid_meo_dataset, stations, meo_stations = load_bj_grid_meo_data(bj_near_stations)
    elif city == "ld" :
        grid_meo_dataset, stations, meo_stations = load_ld_grid_meo_data(ld_near_stations)

    # Supression des doubles
    for station in meo_stations.keys() :
        
        df = meo_stations[station].copy()
        length = df.shape[0]
        order = range(length)
        df['order'] = pd.Series(order, index=df.index)    
        
        
        df["time"] = df.index
        df.set_index("order", inplace=True)
        
        length_1 = df.shape[0]
        
        used_times = []
        for index in df.index :
            time = df.loc[index]["time"]
            if time not in used_times :
                used_times.append(time)
            else : 
                df.drop([index], inplace=True)

        length_2 = df.shape[0]
        delta = length_1 - length_2
        
        df.set_index("time", inplace=True)
        meo_stations[station] = df


    # Complétion des données manquantes

    for station in meo_stations.keys() :
        df = meo_stations[station].copy()
        
        min_time = df.index.min()
        max_time = df.index.max()

        min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
        max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')
        delta_all = max_time - min_time

    for station in meo_stations.keys() :
        df = meo_stations[station].copy()

    delta = datetime.timedelta(hours=1)

    for station in meo_stations.keys() :
        df = meo_stations[station].copy()
        nan_series = pd.Series({key:np.nan for key in df.columns})
        
        min_time = df.index.min()
        max_time = df.index.max()

        min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
        max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')

        time = min_time
        
        while time <=  max_time :
            
            time_str = datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S')
            if time_str not in df.index :
                found_for = False
                i = 0
                while not found_for :
                    i += 1
                    for_time = time - i * delta
                    for_time_str = datetime.date.strftime(for_time, '%Y-%m-%d %H:%M:%S')
                    if for_time_str in df.index :
                        for_row = df.loc[for_time_str]
                        for_step = i
                        found_for = True

                found_back = False
                j = 0
                while not found_back :
                    j += 1
                    back_time = time + j * delta
                    back_time_str = datetime.date.strftime(back_time, '%Y-%m-%d %H:%M:%S')
                    if back_time_str in df.index :
                        back_row = df.loc[back_time_str]
                        back_step = j
                        found_back = True
            
                all_steps = for_step + back_step
            
                if all_steps <= 5 :
                    delata_values = back_row - for_row
                    df.loc[time_str] = for_row + (for_step/all_steps) * delata_values
                else :
                    df.loc[time_str] = nan_series
            
            time += delta
        meo_stations[station] = df


    # Supression des valeurs absurdes du vent

    for station in meo_stations.keys():
        df = meo_stations[station].copy()
        df.replace(999017,0, inplace=True)
        meo_stations[station] = df

    # Finalisation et enregistement des données
    meo_stations_merged = pd.concat(list(meo_stations.values()), axis=1)
    meo_stations_merged.sort_index(inplace=True)

    # meo date go ahead by a day
    meo_stations_merged["date"] = pd.to_datetime(meo_stations_merged.index)
    meo_stations_merged['date'] -= pd.DateOffset(1)
    meo_stations_merged.set_index("date", inplace=True)

    meo_stations_merged.to_csv("prepared_data/%s_meo_data.csv" %(city))