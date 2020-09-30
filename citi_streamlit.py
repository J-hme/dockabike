import streamlit as st
import pandas as pd
import sklearn as sk
import joblib
from datetime import datetime
from pytz import timezone

# # date and time
# tz = timezone('EST')
# datetime.now(tz)
# # Current date time in local system
# date_time_now = datetime.now()
#

# citibike
df_sys_info = pd.read_json('https://gbfs.citibikenyc.com/gbfs/en/system_information.json')
df_station_information = pd.read_json('https://gbfs.citibikenyc.com/gbfs/en/station_information.json') #capacity, name
df_station_status = pd.read_json('https://gbfs.citibikenyc.com/gbfs/en/station_status.json') # online # available

station_iter = df_station_information['data'][0]
stations = []
for j in range(len(station_iter)):
    zipped = zip(['station_name', 'station_id', 'lat', 'lon', 'capacity'], [df_station_information['data'][0][j]['name'],
                df_station_information['data'][0][j]['station_id'],
                df_station_information['data'][0][j]['lat'],
                df_station_information['data'][0][j]['lon'],
                df_station_information['data'][0][j]['capacity']
                              ])
    stations.append(dict(zipped))
stations = pd.DataFrame.from_dict(stations)




#web app

st.write("""
# Dock Right
""")
#st.line_chart(df['riderCount'])
start_station = st.selectbox(
    'Select Start Station',
     stations['station_name'])
'You selected: ', start_station
start_station = stations.loc[stations['station_name'] == 'start_station']
start_station = start_station.astype({'station_id': int})

stop_station = st.selectbox(
    'Select Stop Station',
     stations['station_name'])

'You selected: ', stop_station

stop_station = stations.loc[stations['station_name'] == 'stop_station']
stop_station = stop_station.astype({'station_id': int})




# weather
request = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=NYC&data=tmpf&data=relh&data=feel&data=sped&data=p01i&data=vsby&year1=2020&month1=9&day1=28&year2=2020&month2=9&day2=28&tz=America%2FNew_York&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2'
df_weather = pd.read_csv(f'{request}', na_values=['M', 'T'])
df_weather['p01i'] = df_weather['p01i'].astype(float)
df_weather.fillna(method='ffill')
df_weather['datetime'] = pd.to_datetime(df_weather['valid'])
df_weather['Month'] = df_weather['datetime'].dt.month
df_weather['Day'] = df_weather['datetime'].dt.day
df_weather['Hour'] = df_weather['datetime'].dt.hour
df_weather['Weekday'] = ((df_weather['datetime'].dt.dayofweek) // 5 == 0).astype(float)
df_weather = df_weather.groupby(['Month', 'Day', 'Hour', 'Weekday']).mean().reset_index()
weather = df_weather.tail(1)
weather = weather.fillna(0)

# build input to model
result = pd.concat([weather.reset_index(), start_station.reset_index()], axis=1)
result = result.drop(['index', 'station_name', 'lat', 'lon', 'capacity'], axis=1)
result = result.fillna(0)
X_start = result.reindex(['Month', 'Day', 'Hour', 'Weekday', 'station_id', 'tmpf', 'relh', 'feel', 'sped', 'p01i', 'vsby'], axis=1).values

result = pd.concat([weather.reset_index(), stop_station.reset_index()], axis=1)
result = result.drop(['index', 'station_name', 'lat', 'lon', 'capacity'], axis=1)
result = result.fillna(0)
X_stop = result.reindex(['Month', 'Day', 'Hour', 'Weekday', 'station_id', 'tmpf', 'relh', 'feel', 'sped', 'p01i', 'vsby'], axis=1).values



# model
#filename = 'C:/Users/javad/Documents/Python Scripts/git_folder/final_prediction2.joblib'
filename = 'final_prediction2.joblib'
loaded_model = joblib.load(filename)
y_start = loaded_model.predict(X_start)
y_stop = loaded_model.predict(X_stop)

st.write('Chance of your start being empty (-1) or full (+1): ')
st.write(y_start)

st.write('Chance of your stop being empty (-1) or full (+1): ')
st.write(y_stop)

map_data = pd.DataFrame(
    pd.np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

