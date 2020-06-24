import os
import numpy as np
import pandas as pd
import datetime
import geopy.distance
from matplotlib import pyplot as plt

def load_meo_data(useful_stations, city):
    # Returns 3 objects : 
    # - meo_dataset : the input dataset where the utc_time column has been replaced by a column named time with dates
    #                 instead of strings
    # - stations : a list of all the stations' names
    # - meo_stations a dictionnary where you can find for each station its air quality

    if city == "bj" :
        meo_dataset = pd.read_csv("./data/Beijing_historical_meo_grid.csv")
    elif city == "ld":
        meo_dataset = pd.read_csv("./data/London_historical_meo_grid.csv")
    
    # We prepare the column
    meo_dataset.drop("longitude", axis=1, inplace=True)    
    meo_dataset.drop("latitude", axis=1, inplace=True)

    meo_dataset.rename(index=str, columns={"wind_speed/kph" : "wind_speed"}, inplace=True)

    meo_dataset.sort_index(inplace=True)
    meo_dataset.drop_duplicates(subset=None, inplace=True)

    meo_dataset["time"] = pd.to_datetime(meo_dataset['utc_time'])
    meo_dataset.set_index("time", inplace=True)
    meo_dataset.drop("utc_time", axis=1, inplace=True)

    # Names of all stations (we use set to get unique names)
    stations = set(meo_dataset['stationName'])

    # Dictionnary that gives for each stations the meo data
    meo_stations = {}
    for aq_station_name, meo_station_name in useful_stations.items() :
        if meo_station_name in stations :
            meo_station = meo_dataset[meo_dataset["stationName"]==meo_station_name].copy()
            meo_station.drop("stationName", axis=1, inplace=True)
            if "None" in meo_station.columns :
                meo_station.drop("None", axis=1, inplace=True)

            meteo_feature_names = meo_station.columns.values.tolist()
            column_names = {}
            for meteo_feature_name in meteo_feature_names:
                column_names[meteo_feature_name] = aq_station_name+"_"+meteo_feature_name
            meo_station_renamed = meo_station.rename(index=str, columns=column_names)
            meo_stations[aq_station_name] = meo_station_renamed        

    return meo_dataset, stations, meo_stations

def generate_nearest_stations(city):
    near_stations = {}
    if city == "ld":
        aq_locations = pd.read_csv("./data/London_AirQuality_Stations.csv", usecols=[0,4,5])
        aq_locations.columns = ['station_id','Latitude','Longitude']
        grid_locations = pd.read_csv("./data/London_grid_weather_station.csv",names=['grid_id','Latitude','Longitude'])
    elif city == 'bj':
        aq_locations = pd.read_excel("./data/Beijing_AirQuality_Stations_en.xlsx", skiprows=11, names=['station_id','Latitude','Longitude']).drop([12,13,25,26,34,35])
        grid_locations = pd.read_csv("./data/Beijing_grid_weather_station.csv",names=['grid_id','Longitude','Latitude'])
    for index_aq, row_aq in aq_locations.iterrows():
        max_dist = 1e10
        coords_aq = (row_aq['Longitude'], row_aq['Latitude'])
        for index_grid, row_grid in grid_locations.iterrows():
            coords_grid = (row_grid['Longitude'], row_grid['Latitude'])
            dist = geopy.distance.vincenty(coords_aq, coords_grid)
            if dist < max_dist:
                max_dist = dist
                station_id = row_aq['station_id']
                near_stations[station_id] = row_grid['grid_id']
    return near_stations

def meo_data_preprocess(city="bj"): 
    if city == "bj" :
        grid_meo_dataset, stations, meo_stations = load_meo_data(generate_nearest_stations(city), city)
    elif city == "ld" :
        grid_meo_dataset, stations, meo_stations = load_meo_data(generate_nearest_stations(city), city)

    # Supression of the duplicates
    for station in meo_stations.keys() :
        df = meo_stations[station].copy()
        length = df.shape[0]
        order = range(length)
        df['order'] = pd.Series(order, index=df.index)    
        
        
        df["time"] = df.index
        df.set_index("order", inplace=True)
        
        used_times = []
        for index in df.index :
            time = df.loc[index]["time"]
            if time not in used_times :
                used_times.append(time)
            else : 
                df.drop([index], inplace=True)
        
        df.set_index("time", inplace=True)
        meo_stations[station] = df


    # Completion of the missing data
    for station in meo_stations.keys() :
        df = meo_stations[station].copy()
        nan_series = pd.Series({key:np.nan for key in df.columns})

        min_time = datetime.datetime.strptime(df.index.min(), '%Y-%m-%d %H:%M:%S')
        max_time = datetime.datetime.strptime(df.index.max(), '%Y-%m-%d %H:%M:%S')

        time = min_time
        
        while time <=  max_time :
            
            time_str = datetime.date.strftime(time, '%Y-%m-%d %H:%M:%S')
            if time_str not in df.index :
                found_before = False
                i = 0
                while not found_before :
                    i += 1
                    before_time = time - i * datetime.timedelta(hours=1)
                    before_time_str = datetime.date.strftime(before_time, '%Y-%m-%d %H:%M:%S')
                    if before_time_str in df.index :
                        before_row = df.loc[before_time_str]
                        before_step = i
                        found_before = True

                found_after = False
                j = 0
                while not found_after :
                    j += 1
                    after_time = time + j * datetime.timedelta(hours=1)
                    after_time_str = datetime.date.strftime(after_time, '%Y-%m-%d %H:%M:%S')
                    if after_time_str in df.index :
                        after_row = df.loc[after_time_str]
                        after_step = j
                        found_after = True
            
                combined_steps = before_step + after_step
            
                delata_values = after_row - before_row
                df.loc[time_str] = before_row + (before_step/combined_steps) * delata_values
            
            time += datetime.timedelta(hours=1)
        meo_stations[station] = df

    # We concat the data and save it in a csv file
    meo_stations_merged = pd.concat(list(meo_stations.values()), axis=1)
    if city == 'bj':
        column_names_without_aq = []
        for coln_name in meo_stations_merged.columns :
            column_names_without_aq.append(coln_name.replace('_aq',''))
        meo_stations_merged.columns = column_names_without_aq
    meo_stations_merged.sort_index(inplace=True)
    meo_stations_merged["date"] = pd.to_datetime(meo_stations_merged.index)
    meo_stations_merged.set_index("date", inplace=True)
    meo_stations_merged.to_csv("prepared_data/%s_meo_data.csv" %(city))
    print("Meteo data of %s prepared ！" %(city))