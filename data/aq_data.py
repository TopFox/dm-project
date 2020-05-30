import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import os


def load_bj_aq_data():
    bj_aq_files  = ["./data/beijing_17_18_aq.csv", "./data/beijing_201802_201803_aq.csv"]
    bj_aq_datas = []
    for aq_file in bj_aq_files :
        bj_aq_data = pd.read_csv(aq_file)
        if "id" in bj_aq_data.columns : 
            bj_aq_data.drop("id", axis=1, inplace=True)
        bj_aq_datas.append(bj_aq_data)

    bj_aq_merged_data = pd.concat(bj_aq_datas, ignore_index=True)
    bj_aq_merged_data.sort_index(inplace=True)
    bj_aq_merged_data.drop_duplicates(subset=None, inplace=True)

    bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged = load_city_aq_data(bj_aq_merged_data)

    return bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged

def load_ld_aq_data():
    ld_aq_files  = ["./data/London_historical_aqi_other_stations_20180331.csv", "./data/London_historical_aqi_forecast_stations_20180331.csv"]
    ld_aq_datas = []

    for aq_file in ld_aq_files :
        ld_aq_data = pd.read_csv(aq_file)

        for column_name in ld_aq_data.columns :
            if 'Unnamed:' in column_name : 
                ld_aq_data.drop(column_name, axis=1, inplace=True)

        name_pair = {}

        if 'station_id' in ld_aq_data.columns :
            name_pair['station_id'] = 'stationId'
        elif 'Station_ID' in ld_aq_data.columns :  
            name_pair['Station_ID'] = 'stationId'

        name_pair['MeasurementDateGMT'] = 'utc_time'
        name_pair["PM2.5 (ug/m3)"] = "PM2.5"
        name_pair["PM10 (ug/m3)"] = "PM10"
        name_pair["NO2 (ug/m3)"] = "NO2"

        if "id" in ld_aq_data.columns : 
            ld_aq_data.drop("id", axis=1, inplace=True)

        ld_aq_data.rename(index=str, columns=name_pair, inplace=True)
        ld_aq_datas.append(ld_aq_data)

    ld_aq_merged_data = pd.concat(ld_aq_datas, ignore_index=True)
    ld_aq_merged_data.sort_index(inplace=True)
    ld_aq_merged_data.drop_duplicates(subset=None, inplace=True)

    ld_aq_dataset, stations, ld_aq_stations, ld_aq_stations_merged = load_city_aq_data(ld_aq_merged_data)

    return ld_aq_dataset, stations, ld_aq_stations, ld_aq_stations_merged

def load_city_aq_data(aq_dataset):
	# turn date from string type to datetime type
	aq_dataset["time"] = pd.to_datetime(aq_dataset['utc_time'])
	aq_dataset.set_index("time", inplace=True)
	aq_dataset.drop("utc_time", axis=1, inplace=True)

	aq_dataset = aq_dataset[pd.isnull(aq_dataset.stationId) != True]

	# names of all stations
	stations = set(aq_dataset['stationId'])

	# a dict of station aq
	aq_stations = {}
	for station in stations:
		aq_station = aq_dataset[aq_dataset["stationId"]==station].copy()
		aq_station.drop("stationId", axis=1, inplace=True)

		# rename
		original_names = aq_station.columns.values.tolist()
		names_dict = {original_name : station+"_"+original_name for original_name in original_names}
		aq_station.rename(index=str, columns=names_dict, inplace=True)
		aq_station.drop_duplicates(inplace=True)
		aq_stations[station] = aq_station

	aq_stations_merged = pd.concat(list(aq_stations.values()), axis=1)

	length = aq_stations_merged.shape[0]
	order = range(length)
	aq_stations_merged['order'] = pd.Series(order, index=aq_stations_merged.index)

	return aq_dataset, stations, aq_stations, aq_stations_merged

def aq_data_preparation(city):
    # On load les données de la ville choisie
    if city == "bj" :
        aq_data, stations, aq_stations, aq_stations_merged = load_bj_aq_data()
    elif city == "ld" :
        aq_data, stations, aq_stations, aq_stations_merged = load_ld_aq_data()

    # On supprime les valeurs en double
    aq_stations_merged["time"] = aq_stations_merged.index
    aq_stations_merged.set_index("order", inplace=True)
    present_times = []
    for index in aq_stations_merged.index :
        time = aq_stations_merged.loc[index]["time"]
        if time not in present_times :
            present_times.append(time)
        else : 
            aq_stations_merged.drop([index], inplace=True)
    aq_stations_merged.set_index("time", inplace=True)

    # 3. 缺失值的分析 Analyse des valeurs manquantes
    min_time = aq_stations_merged.index.min()
    max_time = aq_stations_merged.index.max()

    min_time = datetime.datetime.strptime(min_time, '%Y-%m-%d %H:%M:%S')
    max_time = datetime.datetime.strptime(max_time, '%Y-%m-%d %H:%M:%S')

    # 3.1 整小时缺失 L'heure manque

    delta = datetime.timedelta(hours=1)
    time = min_time
    missing_hours = []
    missing_hours_str = []


    while time <=  max_time :
        if datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S') not in aq_stations_merged.index :
            missing_hours.append(time)
            missing_hours_str.append(datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S'))
        time += delta

    # 3.2 某个小时某个站点数据缺失 On a l'heure mais pas les données
    if city == "bj" :
        aq_station_locations = pd.read_excel("./data/Beijing_AirQuality_Stations_en.xlsx", header=10, usecols="A:C",skiprows=[11]).drop([12, 13, 25, 26, 34, 35]).reset_index(drop=True).rename(columns={"Station ID": "stationName"})
    elif city == "ld" :
        aq_station_locations = pd.read_csv("./data/London_AirQuality_Stations.csv")
        aq_station_locations = aq_station_locations[["Unnamed: 0", "Latitude", "Longitude"]]
        aq_station_locations.rename(index=str, columns={"Unnamed: 0":"stationName", "Latitude":"latitude", "Longitude":"longitude"}, inplace=True)
    #  Pour une station de qualité de l'air, organisez les autres stations en fonction de la distance de la station et enregistrez-les sous forme de liste

    for index_t in aq_station_locations.index:
        row_t = aq_station_locations.loc[index_t]
        # location of target station
        long_t = row_t["longitude"]
        lati_t = row_t["latitude"]
        # column name
        station_name = row_t["stationName"]
        
        # add a new column to df
        all_dis = []
        for index in aq_station_locations.index:
            row = aq_station_locations.loc[index]
            long = row['longitude']
            lati = row['latitude']
            dis = np.sqrt((long-long_t)**2 + (lati-lati_t)**2)
            all_dis.append(dis)
        
        aq_station_locations[station_name] = all_dis

    # 以每一个站的名字为 key，以其他站的名字组成的列表为 value list，列表中从前向后距离越来越远
    near_stations = {}
    for index_t in aq_station_locations.index:
        target_station_name = aq_station_locations.loc[index_t]['stationName']
        ordered_stations_names = aq_station_locations.sort_values(by=target_station_name)['stationName'].values[1:]
        near_stations[target_station_name] = ordered_stations_names

    def get_estimated_value(station_name, feature_name, near_stations, row):
        near_stations = near_stations[station_name]    # A list of nearest stations
        for station in near_stations :                 # 在最近的站中依次寻找非缺失值
            feature = station + "_" +feature_name
            if not pd.isnull(row[feature]):
                return row[feature]
            
        return 0

    for index in aq_stations_merged.index :
        row = aq_stations_merged.loc[index].copy()
        for feature in row.index :
            # print(feature)
            if pd.isnull(row[feature]) :
                elements = feature.split("_")                  
                if city == "bj" :                                  # feature example： nansanhuan_aq_PM2.5
                    station_name = elements[0] + "_" + elements[1] # nansanhuan_aq
                    feature_name = elements[2]                     # PM2.5
                elif city == "ld" :                                # feature example： KC1_NO2 (ug/m3)
                    station_name = elements[0]                     # KC1
                    feature_name = elements[1]                     # NO2 (ug/m3)
                row[feature] = get_estimated_value(station_name, feature_name, near_stations, row)
        aq_stations_merged.loc[index] = row

    assert (pd.isnull(aq_stations_merged).any().any()) == False, "数据中还有缺失值(局部处理后)"

    # London 并不是每个站点都有用
    if city == "ld" :
        
        stations_to_predict = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4']
        features = ['NO2', 'PM10', 'PM2.5']

        all_features = []
        for station in stations_to_predict :
            for feature in features:
                station_feature = station + "_" + feature
                all_features.append(station_feature)

        aq_stations_merged = aq_stations_merged[all_features]
    # 3.3 整小时的缺失的处理
    keep_hours = []
    drop_hours = []


    # 先对小于5小时的进行填充
    delta = datetime.timedelta(hours=1)

    for hour in missing_hours_str : 
        
        time = datetime.datetime.strptime(hour, '%Y-%m-%d %H:%M:%S')
        
        # 前边第几个是非空的
        found_for = False
        i = 0
        while not found_for :
            i += 1
            for_time = time - i * delta
            for_time_str = datetime.date.strftime(for_time, '%Y-%m-%d %H:%M:%S')
            if for_time_str in aq_stations_merged.index :
                for_row = aq_stations_merged.loc[for_time_str]
                for_step = i
                found_for = True
                
                
        # 后边第几个是非空的
        found_back = False
        j = 0
        while not found_back :
            j += 1
            back_time = time + j * delta
            back_time_str = datetime.date.strftime(back_time, '%Y-%m-%d %H:%M:%S')
            if back_time_str in aq_stations_merged.index :
                back_row = aq_stations_merged.loc[back_time_str]
                back_step = j
                found_back = True
        
        # print(for_step, back_step)
        all_steps = for_step + back_step
        if all_steps > 5 :
            drop_hours.append(hour)
        else : 
            keep_hours.append(hour)
            # 插值
            delata_values = back_row - for_row
            aq_stations_merged.loc[hour] = for_row + (for_step/all_steps) * delata_values        


    # 再对超过5小时的填充 NAN
    nan_series = pd.Series({key:np.nan for key in aq_stations_merged.columns})

    for hour in drop_hours:
        aq_stations_merged.loc[hour] = nan_series

    aq_stations_merged.sort_index(inplace=True)

    # 3.4 数据存储

    aq_stations_merged.to_csv("./prepared_data/%s_aq_data.csv" %(city))
    
    print("Traitement des données de la ville de %s terminé ！" %(city))