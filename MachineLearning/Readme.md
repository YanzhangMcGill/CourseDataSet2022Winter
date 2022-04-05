# NYC Manhattan dataset

The data in a (7 km * 12km) rectangular area is used. Each region is a 1km*1km square in the map, thus the grid map : (7, 12). The map is aligned along Manhattan. And the unit for ```x``` and ```y``` is 1 km.

> ```boxed-202106-citibike-tripdata.csv```: filtered data, data format: (ride_id, datetime, station_id, x, y, boxed_region).
>
> ```station_info.csv```: implies what stations a service_region will contain.
>
> ```demand-boxed-202106.csv```: panel data, the demand at each service region per 1 hours.