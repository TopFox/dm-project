import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import os


def load_bj_aq_data():
    # Loads air quality data of Beijing through two files and calls load_city_aq_data() on it
    bj_aq_files  = ["./data/beijing_17_18_aq.csv", "./data/beijing_201802_201803_aq.csv"]
    bj_aq_datas = []
    for aq_file in bj_aq_files :
        bj_aq_data = pd.read_csv(aq_file)
        if "id" in bj_aq_data.columns : 
            bj_aq_data.drop("id", axis=1, inplace=True)
        if "NO2" in bj_aq_data.columns : 
            bj_aq_data.drop("NO2", axis=1, inplace=True)
        if "SO2" in bj_aq_data.columns : 
            bj_aq_data.drop("SO2", axis=1, inplace=True)
        if "CO" in bj_aq_data.columns : 
            bj_aq_data.drop("CO", axis=1, inplace=True)
        bj_aq_datas.append(bj_aq_data)

    bj_aq_merged_data = pd.concat(bj_aq_datas, ignore_index=True)
    bj_aq_merged_data.sort_index(inplace=True)
    bj_aq_merged_data.drop_duplicates(subset=None, inplace=True)

    bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged = load_city_aq_data(bj_aq_merged_data)

    return bj_aq_dataset, stations, bj_aq_stations, bj_aq_stations_merged

def load_ld_aq_data():
    # Loads air quality data of London through two files and calls load_city_aq_data() on it
    ld_aq_files  = ["./data/London_historical_aqi_other_stations_20180331.csv", "./data/London_historical_aqi_forecast_stations_20180331.csv"]
    ld_aq_datas = []

    for aq_file in ld_aq_files :
        ld_aq_data = pd.read_csv(aq_file)
       
        for column_name in ld_aq_data.columns :
            if 'Unnamed:' in column_name : 
                ld_aq_data.drop(column_name, axis=1, inplace=True)
            if 'NO2 (ug/m3)' in column_name :
                ld_aq_data.drop(column_name, axis=1, inplace=True)

         # We need the names of the column to be the same for London and Beijing data
        adapted_name_pairs = {}

        if 'station_id' in ld_aq_data.columns :
            adapted_name_pairs['station_id'] = 'stationId'
        elif 'Station_ID' in ld_aq_data.columns :  
            adapted_name_pairs['Station_ID'] = 'stationId'

        adapted_name_pairs['MeasurementDateGMT'] = 'utc_time'
        adapted_name_pairs["PM2.5 (ug/m3)"] = "PM2.5"
        adapted_name_pairs["PM10 (ug/m3)"] = "PM10"

        if "id" in ld_aq_data.columns : 
            ld_aq_data.drop("id", axis=1, inplace=True)

        ld_aq_data.rename(index=str, columns=adapted_name_pairs, inplace=True)
        ld_aq_datas.append(ld_aq_data)

    ld_aq_merged_data = pd.concat(ld_aq_datas, ignore_index=True)
    ld_aq_merged_data.sort_index(inplace=True)
    ld_aq_merged_data.drop_duplicates(subset=None, inplace=True)

    ld_aq_dataset, stations, ld_aq_stations, ld_aq_stations_merged = load_city_aq_data(ld_aq_merged_data)

    return ld_aq_dataset, stations, ld_aq_stations, ld_aq_stations_merged

def load_city_aq_data(aq_dataset):
    # Returns 4 objects : 
    # - aq_dataset : the input dataset where the utc_time column has been replaced by a column named time with dates
    #                instead of strings
    # - stations : a list of all the stations' names
    # - aq_stations a dictionnary where you can find for each station its air quality
    # - aq_stations_merged : dataframe with the air quality of each cities

    # We need a real date and not the just a string
    aq_dataset["time"] = pd.to_datetime(aq_dataset['utc_time'])
    aq_dataset.set_index("time", inplace=True)
    aq_dataset.drop("utc_time", axis=1, inplace=True)

    aq_dataset = aq_dataset[pd.isnull(aq_dataset.stationId) != True]

    # Names of all stations (we use set to get unique names)
    stations = set(aq_dataset['stationId'])

    # a dict of station aq
    aq_stations = {}
    for station in stations:
        aq_station = aq_dataset[aq_dataset["stationId"]==station].copy()
        aq_station.drop("stationId", axis=1, inplace=True)

        # rename
        pollution_names = aq_station.columns.values.tolist()
        column_names = {}
        for pollution_name in pollution_names :
            column_names[pollution_name] = station+"_"+pollution_name
        aq_station.rename(index=str, columns=column_names, inplace=True)
        aq_station.drop_duplicates(inplace=True)
        aq_stations[station] = aq_station

    aq_stations_merged = pd.concat(list(aq_stations.values()), axis=1)

    length = aq_stations_merged.shape[0]
    order = range(length)
    aq_stations_merged['order'] = pd.Series(order, index=aq_stations_merged.index)

    return aq_dataset, stations, aq_stations, aq_stations_merged

def get_feature_value_of_nearest_city(station_name, feature_name, near_stations, row):
    near_stations = near_stations[station_name]
    for station in near_stations :
        feature = station + "_" +feature_name
        if not pd.isnull(row[feature]):
            return row[feature]
        
    return 0

def aq_data_preparation(city):
    # We load the data of the chosen city
    if city == "bj" :
        aq_data, stations, aq_stations, aq_stations_merged = load_bj_aq_data()
    elif city == "ld" :
        aq_data, stations, aq_stations, aq_stations_merged = load_ld_aq_data()
    else:
        print('There was an error while preparing the air quality data : city not found.')

    # Supression of the repetitions
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

    # We extract the locations of the stations
    if city == "bj" :
        aq_station_locations = pd.read_excel("./data/Beijing_AirQuality_Stations_en.xlsx", header=10, usecols="A:C",skiprows=[11]).drop([12, 13, 25, 26, 34, 35]).reset_index(drop=True).rename(columns={"Station ID": "stationName"})
    elif city == "ld" :
        aq_station_locations = pd.read_csv("./data/London_AirQuality_Stations.csv")
        aq_station_locations = aq_station_locations[["Unnamed: 0", "Latitude", "Longitude"]]
        aq_station_locations.rename(index=str, columns={"Unnamed: 0":"stationName", "Latitude":"latitude", "Longitude":"longitude"}, inplace=True)
    
    # For each station, we put in an array the distances from each other station
    for index_t in aq_station_locations.index:
        row_t = aq_station_locations.loc[index_t]
        long_t = row_t["longitude"]
        lati_t = row_t["latitude"]
        station_name = row_t["stationName"]
        
        distances = []
        for index in aq_station_locations.index:
            row = aq_station_locations.loc[index]
            long = row['longitude']
            lati = row['latitude']
            dis = np.sqrt((long-long_t)**2 + (lati-lati_t)**2)
            distances.append(dis) 
        aq_station_locations[station_name] = distances

    # We create a dictionnary that gives for each stations the nearest cities ordered
    near_stations = {}
    for index_t in aq_station_locations.index:
        target_station_name = aq_station_locations.loc[index_t]['stationName']
        ordered_stations_names = aq_station_locations.sort_values(by=target_station_name)['stationName'].values[1:]
        near_stations[target_station_name] = ordered_stations_names

    # For each station where the hours aren't missing but at least a feature is, we get the feature value of the closest station
    for index in aq_stations_merged.index :
        row = aq_stations_merged.loc[index].copy()
        for feature in row.index :
            if pd.isnull(row[feature]) :
                elements = feature.split("_")                  
                if city == "bj" :                                  
                    station_name = elements[0] + "_" + elements[1]
                    feature_name = elements[2]                     
                elif city == "ld" :                                
                    station_name = elements[0]                     
                    feature_name = elements[1]                     
                row[feature] = get_feature_value_of_nearest_city(station_name, feature_name, near_stations, row)
        aq_stations_merged.loc[index] = row

    # For london, we drop the useless stations
    if city == "ld" :     
        stations_to_predict = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4']
        features = ['PM10', 'PM2.5']

        all_features = []
        for station in stations_to_predict :
            for feature in features:
                station_feature = station + "_" + feature
                all_features.append(station_feature)

        aq_stations_merged = aq_stations_merged[all_features]

    drop_hours = []

    # We store the missinbg hours in an array
    time = datetime.datetime.strptime(aq_stations_merged.index.min(), '%Y-%m-%d %H:%M:%S')
    missing_times = []
    while time <= datetime.datetime.strptime(aq_stations_merged.index.max(), '%Y-%m-%d %H:%M:%S') :
        if datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S') not in aq_stations_merged.index :
            missing_times.append(datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S'))
        time += datetime.timedelta(hours=1)

    for hour in missing_times : 
        
        time = datetime.datetime.strptime(hour, '%Y-%m-%d %H:%M:%S')
        
        # We find the closest value before the missing hour
        found_before = False
        i = 0
        while not found_before :
            i += 1
            before_time = datetime.date.strftime(time + i * datetime.timedelta(hours=1), '%Y-%m-%d %H:%M:%S')
            if before_time in aq_stations_merged.index :
                combined_step = aq_stations_merged.loc[before_time]
                before_step = i
                found_before = True
                
                
        # We find the closest value after the missing hour
        found_after = False
        j = 0
        while not found_after :
            j += 1
            after_time = datetime.date.strftime(time - j * datetime.timedelta(hours=1), '%Y-%m-%d %H:%M:%S')
            if after_time in aq_stations_merged.index :
                after_row = aq_stations_merged.loc[after_time]
                after_step = j
                found_after = True
        
        combined_row = before_step + after_step

        # If the combined steps are more than 5, we drop the hours, ele we inteprolate
        if combined_row > 5 :
            drop_hours.append(hour)
        else :
            delata_values = after_row - combined_step
            aq_stations_merged.loc[hour] = combined_step + (before_step/combined_row) * delata_values        

    # For each hour that we need to drop, we replace the whole line with a serie of nan values
    for hour in drop_hours:
        aq_stations_merged.loc[hour] = pd.Series({key:np.nan for key in aq_stations_merged.columns})

    aq_stations_merged.sort_index(inplace=True)

    if city == 'bj':
        column_names_without_aq = []
        for coln_name in aq_stations_merged.columns :
            column_names_without_aq.append(coln_name.replace('_aq',''))
        aq_stations_merged.columns = column_names_without_aq
    # We save the cleaned data in a csv file
    aq_stations_merged.to_csv("./prepared_data/%s_aq_data.csv" %(city))
    print("Air quality data of %s prepared ！" %(city))