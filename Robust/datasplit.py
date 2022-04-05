import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,timedelta,date
from collections import Counter
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

path = './data/'
data_name = '202106-citibike-tripdata.csv'
downloaddata = pd.read_csv(path+data_name)
downloaddata['datetime'] = pd.to_datetime(downloaddata['started_at'])
sorted_data = downloaddata.sort_values(by='datetime').reset_index(drop=True)
select_col = ['ride_id','datetime','start_station_id','start_lat','start_lng']
data_df = sorted_data[select_col]
data_df = data_df.dropna().reset_index(drop=True)

# visualize the data in a map
BBox = (data_df.start_lng.min(),   data_df.start_lng.max(),      
         data_df.start_lat.min(), data_df.start_lat.max())

start_time = datetime(2021,6,1,0,0,0)
end_time = datetime(2021,6,1,12,0,0)
data_firstday = data_df[(data_df['datetime']<end_time)&(data_df['datetime']>start_time)]

ruh_m = plt.imread('NYC_map.png')
plt.figure(figsize=(10,22.4))
ax = plt.axes()
ax.scatter(data = data_firstday, x = 'start_lng', y = 'start_lat', zorder=1, alpha=0.15,c='forestgreen',edgecolors='black',s=30,linewidths=0.3)
ax.set_title('Plotting Spatial Data Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'auto')
plt.savefig('bike_map_{}.jpg'.format(start_time.date()), format='jpg', transparent=False, dpi=300, pad_inches = 0)

# plot the demand in a station
tmp_df = data_df[data_df['start_station_id']==6912.01]
Start_Time = datetime(2021,6,1,0,0,0)
End_Time = datetime(2021,7,1,0,0,0)
step = timedelta(hours=1)
start_time = Start_Time
datetime_list = []
density_list = []
while start_time<End_Time:
    end_time = start_time+step
    print('Time: {}'.format(start_time))
    tmp_step_df = tmp_df[(tmp_df['datetime']>start_time)&(tmp_df['datetime']<=end_time)]
    datetime_list.append(start_time)
    density_list.append(len(tmp_step_df))
    start_time = end_time
density_df = pd.DataFrame({'datetime':datetime_list,'density':density_list})
plt.plot('datetime','density',data=density_df[density_df['datetime']<datetime(2021,6,3,1,0,0)])
plt.show()

# filter data in manhattan
nybb = gpd.read_file('content/geo_export_c2e49df8-2906-4a44-be12-b5c7f339425c.shp')
Manhattan_geometry = nybb[nybb['boro_name']=='Manhattan']['geometry'].to_list()[0]
# designate coordinate system
crs = {'init': 'epsg:4326'}
station_df = data_df[data_df.duplicated(subset='start_station_id', keep='first')==False][['start_station_id','start_lat','start_lng']].reset_index(drop=True)
# zip x and y coordinates into single feature
geometry = [Point(xy) for xy in zip(station_df['start_lng'], station_df['start_lat'])]
# create GeoPandas dataframe
geo_df = gpd.GeoDataFrame(station_df,crs = crs,geometry = geometry)
geo_df = geo_df.assign(**{'in_manhattan': geo_df.within(Manhattan_geometry)})
to_delete_list = [4590.01,4440.02,4374.01,6566.01,6599.01,6814.01,7327.01,7514.01,7571.01,7618.01]
geo_df.loc[geo_df['start_station_id'].isin(to_delete_list),'in_manhattan']=False
in_manhattan_df = geo_df[['start_station_id','in_manhattan']]
data_df_bool_in_manhattan = data_df.merge(in_manhattan_df,how='left',on='start_station_id')
Manhattan_data_df = data_df_bool_in_manhattan[data_df_bool_in_manhattan['in_manhattan']].reset_index(drop=True)
# apply k-means to cluster
Manhattan_station_df = geo_df[['start_station_id','start_lat','start_lng','in_manhattan']][geo_df['in_manhattan']].reset_index(drop=True)
Manhattan_station_location = Manhattan_station_df[['start_lat','start_lng']].to_numpy()
kmeans = KMeans(n_clusters=50, random_state=1234).fit(Manhattan_station_location)
labels_ = kmeans.labels_
Manhattan_station_df['service_region'] = kmeans.labels_
Manhattan_station_df = Manhattan_station_df.sort_values(by='service_region').reset_index(drop=True)
Manhattan_station_df.to_csv(path+'Manhattan_service_region_design.csv',index=0)
Manhattan_data_df = Manhattan_data_df.merge(Manhattan_station_df[['start_station_id','service_region']],how='left',on='start_station_id')
Manhattan_data_df.to_csv(path+'manhattan-'+data_name,index=0)

# draw service region
centers = kmeans.cluster_centers_[:, [1, 0]]
vor = Voronoi(centers)
ruh_m = plt.imread('NYC_map.png')
alpha_manhattan = plt.imread('Manhattan_alphas.png')
plt.figure(figsize=(10,22.4))
ax = plt.axes()
ax.scatter(data = Manhattan_station_df, x = 'start_lng', y = 'start_lat',  alpha=0.5,c=Manhattan_station_df['service_region'],edgecolors='black',s=100,linewidths=0.3)
voronoi_plot_2d(vor,ax = ax, show_vertices=False,line_colors='black',
                line_width=2, line_alpha=0.6, point_size=2)
ax.set_title('Plotting Spatial Data Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, extent = BBox, aspect= 'auto')
ax.imshow(alpha_manhattan ,zorder=100, extent = BBox, aspect= 'auto')
plt.savefig('bike_map_Manhattan.jpg', format='jpg', transparent=False, dpi=300, pad_inches = 0)

# aggregate data
Manhattan_data_df = pd.read_csv(path+'manhattan-'+data_name)
Manhattan_data_df['datetime'] = pd.to_datetime(Manhattan_data_df['datetime'])
service_region_num = 50
Start_Time = datetime(2021,6,1,0,0,0)
End_Time = datetime(2021,7,1,0,0,0)
step = timedelta(hours=1)
start_time = Start_Time
datetime_list = []
region_list = []
demand_list = []
while start_time<End_Time:
    end_time = start_time+step
    print('Time: {}'.format(start_time))
    tmp_step_df = Manhattan_data_df[(Manhattan_data_df['datetime']>start_time)&(Manhattan_data_df['datetime']<=end_time)]
    for region_id in range(service_region_num):
        tmp_step_df_one_region = tmp_step_df[tmp_step_df['service_region']==region_id]
        demand_list.append(len(tmp_step_df_one_region))
    datetime_list.extend([start_time]*service_region_num)
    region_list.extend(list(range(service_region_num)))
    start_time = end_time
demand_df = pd.DataFrame({'datetime':datetime_list,'service_region':region_list,'demand':demand_list})
demand_df.to_csv(path+'demand-manhattan-202106.csv',index=0)

