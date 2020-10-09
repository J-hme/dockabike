import streamlit as st
import pandas as pd
import joblib

# In this section I read the static and real-time data from cibike API


@st.cache
def load_stations():
    # This read citibike static information: capacity, name, id, lat and lon of each station and name of the station
    df_station_information = pd.read_json('https://gbfs.citibikenyc.com/gbfs/en/station_information.json')
    # read all the stations and store in data frame : stations
    station_iter = len(df_station_information['data'][0])
    station = []
    for j in range(station_iter):
        zipped = zip(['station_name', 'station_id', 'lat', 'lon', 'capacity'],
                     [df_station_information['data'][0][j]['name'],
                      df_station_information['data'][0][j]['station_id'],
                      df_station_information['data'][0][j]['lat'],
                      df_station_information['data'][0][j]['lon'],
                      df_station_information['data'][0][j]['capacity']
                      ])
        station.append(dict(zipped))

    station = pd.DataFrame.from_dict(station)
    return station


@st.cache
def load_distances():
    df_dist = pd.read_csv('distances.csv')
    return df_dist


@st.cache
def load_station_realtime():
    df_st_status = pd.read_json('https://gbfs.citibikenyc.com/gbfs/en/station_status.json') # online # available
    return df_st_status


@st.cache
def realtime_weather():
    request = 'https://mesonet.agron.iastate.edu/cgi-bin/request/' \
              'asos.py?station=NYC&data=tmpf&data=relh&data=feel&data=' \
              'sped&data=p01i&data=vsby&year1=2020&month1=9&day1=28&year2=' \
              '2020&month2=9&day2=28&tz=America%2FNew_York&format=onlycomma&latlon=' \
              'no&missing=M&trace=T&direct=no&report_type=1&report_type=2'
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
    return weather


def id_to_name(ids, station):
    return station[station.station_id == ids].station_name.to_list()[0]


def find_realtime_status(all_stations, selected_station):
    status = all_stations['data'][0]
    status = next(item for item in status if item['station_id'] == selected_station)
    return status


def nearby_stations(distances, all_stations, selected_station):
    nearby_ids = distances.sort_values(by=selected_station)['station_id'].astype(str)[1:6].to_list()
    nearby_dis = distances.sort_values(by=selected_station)[selected_station][1:6].to_list()
    nearby_names = [id_to_name(ids, all_stations) for ids in nearby_ids]
    return nearby_ids, nearby_names, nearby_dis


def make_prediction(weather, all_stations, selected_station):
    station = all_stations.loc[all_stations['station_name'] == selected_station]
    station = station.astype({'station_id': int})
    result = pd.concat([weather.reset_index(), station.reset_index()], axis=1)
    result = result.drop(['index', 'station_name', 'lat', 'lon', 'capacity'], axis=1)
    result = result.fillna(0)
    model_input = result.reindex(
        ['Month', 'Day', 'Hour', 'Weekday', 'station_id', 'tmpf', 'relh', 'feel', 'sped', 'p01i', 'vsby'],
        axis=1).values
    filename = 'rfc_2020.joblib'
    model = joblib.load(filename)
    model_output = model.predict(model_input)
    return model_output

def station_reliability_start(free_units, flow):

    if (flow < 0 and abs(flow) <= 0.5 * free_units):
        reliability = 'good to pick up bikes'
    elif (flow < 0 and abs(flow) > .5 * free_units and abs(flow) <= .8 * free_units):
        reliability = 'running out of bikes'
    elif (flow < 0 and abs(flow) > .8 * free_units) or (free_units < 2):
        reliability = 'running out fast'
    else:
        reliability = 'good to pick up bikes'
    return reliability


def station_reliability_stop(free_units, flow):
    if (flow > 0 and abs(flow) <= .5 * free_units):
        reliability = 'good to dock in'
    elif (flow > 0 and abs(flow) > .5 * free_units and abs(flow) <= .8 * free_units):
        reliability = 'docks filling up'
    elif (flow > 0 and abs(flow) > .8 ) or (free_units < 1):
        reliability = 'filling up fast'
    else:
        reliability = 'good to dock in'
    return reliability


# streamlit web app title


st.write("""# Dock Right NY!""")

# load the stations names, capacity, lat and lon
stations = load_stations() # , df_station_status, df_dist
# streamlit web app: create a crop down menu with list of the stations to select
start_station_name = st.selectbox('Select Pick Up Station', stations['station_name'])
# The user selects station name now I need to obtain station_id
start_station = stations.loc[stations['station_name'] == start_station_name]
start_station_id = str(start_station['station_id'].to_list()[0])
# Now that I have tne station ID I can find the real-time number of available bikes and docks
df_station_status = load_station_realtime()
start_status = find_realtime_status(df_station_status, start_station_id)

'Pick up station has: ', start_status['num_bikes_available'], 'free bikes.'

# Now estimate flow and calculate reliability
current_weather = realtime_weather()
pred_sel_start = make_prediction(current_weather, stations, start_station_name)
start_reliablity = station_reliability_start(start_status['num_bikes_available'], pred_sel_start)
'This station is:', start_reliablity, 'within the next hour'


if start_reliablity != 'good to pick up bikes':
    # Now find available bikes and docks in 5 nearby stations
    df_distances = load_distances()
    start_near_ids, start_near_names, start_near_dist = nearby_stations(df_distances, stations, start_station_id)
    start_near_status = [find_realtime_status(df_station_status, ind_st)['num_bikes_available'] for ind_st in
                         start_near_ids]
    # Now estimate the inflow based on historic data and current weather
    pred = [make_prediction(current_weather, stations, station_name) for station_name in start_near_names]
    reliab = []
    for i in range(len(pred)):
        reliab.append(station_reliability_start(pred[i], start_near_status[i]))
    show = pd.DataFrame()
    show['station'] = start_near_names
    show['Distance (m)'] = [round(x * 1000) for x in start_near_dist]
    show['Realtime Free Bikes'] = start_near_status
    show['Reliability'] = reliab#pred
    st.write('Alternative Pick up Stations:', show)



# streamlit web app get the stop station from user
stop_station_name = st.selectbox('Select Drop Off Station', stations['station_name'])
# The user selects station name now I need to obtain station_id
stop_station = stations.loc[stations['station_name'] == stop_station_name]
stop_station_id = str(stop_station['station_id'].to_list()[0])
# Now that I have tne station ID I can find the real-time number of available bikes and docks
stop_status = find_realtime_status(df_station_status, stop_station_id)
'Drop off station has: ', stop_status['num_docks_available'], 'free docks.' \
# Now estimate flow and calculate reliability
pred_sel_stop = make_prediction(current_weather, stations, stop_station_name)
stop_reliability = station_reliability_stop(stop_status['num_docks_available'], pred_sel_stop)
'This station is:', stop_reliability, 'within the next hour'

if stop_reliability != 'good to dock in':
    df_distances = load_distances()
    # Now find available bikes and docks in 5 nearby stations
    stop_near_ids, stop_near_names, stop_near_dist = nearby_stations(df_distances, stations, stop_station_id)
    stop_near_status = [find_realtime_status(df_station_status, ind_st)['num_docks_available'] for ind_st in stop_near_ids]

    # Now estimate the inflow based on historic data and current weather
    pred_stop = [make_prediction(current_weather, stations, station_name) for station_name in stop_near_names]
    reliab_stop = []
    for i in range(len(pred_stop)):
        reliab_stop.append(station_reliability_stop(pred_stop[i], stop_near_status[i]))

    show2 = pd.DataFrame()
    show2['station'] = stop_near_names
    show2['Distance (m)'] = [round(x * 1000) for x in stop_near_dist]
    show2['Realtime Free Docks'] = stop_near_status
    show2['Reliability'] = reliab_stop#pred_stop
    st.write('Alternative Drop off Stations:', show2)

map_data = pd.concat([start_station[['lat', 'lon']], stop_station[['lat', 'lon']]])
st.map(map_data)

