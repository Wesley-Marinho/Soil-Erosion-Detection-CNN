import xarray as xr
import pandas as pd
import numpy as np

def merge_data(data_list):
    """
    Merge a list of xarray datasets based on their coordinates.

    Parameters
    ----------
    data_list : list
        A list of xarray datasets to be merged.

    Returns
    -------
    merged_data : xarray.Dataset
        The merged xarray dataset.

    """
    merged_data = xr.open_mfdataset(data_list, combine='by_coords')
    return merged_data


def rawData(var2get_xr, var_name2get, lat_lon_list):
    """
    Extract data from an xarray dataset based on a list of latitude and longitude coordinates.

    Parameters
    ----------
    var2get_xr : xarray.Dataset
        The xarray dataset containing the data to be extracted.
    var_name2get : str
        The name of the variable to be extracted from the xarray dataset.
    lat_lon_list : str
        A csv file containing a list of latitude and longitude coordinates, with one coordinate per line.
        The first column of the csv file should contain the latitude coordinates, and the second column should contain the longitude coordinates.

    Returns
    -------
    var_ar : numpy.ndarray
        The extracted data as a numpy array.

    """
    df_lat_lon = pd.read_csv(lat_lon_list)
    latitude = df_lat_lon.iloc[:, 1]
    longitude = df_lat_lon.iloc[:, 2]

    return var2get_xr[var_name2get].sel(
        longitude=xr.DataArray(longitude, dims='z'),
        latitude=xr.DataArray(latitude, dims='z'),
        method='nearest').values


def rain_data_calc(lat_lon_list, var2get_yr):
    """
    Calculate the rainfall data for a list of latitude and longitude coordinates from an xarray dataset.

    Parameters
    ----------
    lat_lon_list : str
        A csv file containing a list of latitude and longitude coordinates, with one coordinate per line.
        The first column of the csv file should contain the latitude coordinates, and the second column should contain the longitude coordinates.
    var2get_yr : xarray.Dataset
        The xarray dataset containing the input data.

    Returns
    -------
    df_list : list
        A list of pandas dataframes containing the rainfall data for each latitude and longitude coordinate.

    """
    df_lat_lon = pd.read_csv(lat_lon_list)

    latitude = df_lat_lon.iloc[:, 1]
    longitude = df_lat_lon.iloc[:, 2]

    var_names = ['pr']
    var_name2get = 'pr'

    var2get_xr = xr.combine_by_coords([var2get_yr])
    var_ar = rawData(var2get_xr, var_name2get, lat_lon_list)
    time = var2get_xr.time.values

    df_list = []

    for n in range(len(latitude)):
        if not np.isnan(var_ar[0, n]):
            file = var_ar[:, n::len(longitude)]

            df = pd.DataFrame(file, index=time, columns=var_names)
            df['TIME'] = time
            df['LATITUDE'] = latitude[n]
            df['LONGITUDE'] = longitude[n]
            df_list.append(df)

    return df_list


def media_rain(data_list: list, lat_lon_list: str) -> list:
    """
    Calculate the average rainfall for a list of latitude and longitude coordinates from a list of xarray datasets.

    Parameters
    ----------
    data_list : list
        A list of xarray datasets containing the input data.
    lat_lon_list : str
        A csv file containing a list of latitude and longitude coordinates, with one coordinate per line.
        The first column of the csv file should contain the latitude coordinates, and the second column should contain the longitude coordinates.

    Returns
    -------
    rain_list : list
        A list of pandas dataframes containing the average rainfall for each latitude and longitude coordinate.

    """
    rain_data = merge_data(data_list)

    data_r = rain_data_calc(lat_lon_list, rain_data)

    rain_list = []

    for df in data_r:
        if len(df.columns) == 4:
            df.columns = ['chuva', 'time', 'LATITUDE', 'LONGITUDE']
            df['data_datetime'] = pd.to_datetime(df['time'])
            df['MES'] = df['data_datetime'].dt.month
            df['ANO'] = df['data_datetime'].dt.year
            df = df.drop(["time", "data_datetime"], axis=1)

            acumulado_list = df.groupby(['ANO', 'MES', 'LATITUDE', 'LONGITUDE'])['chuva'].sum().reset_index()
            acumulado_list['chuva'] = acumulado_list['chuva'].round(4)

            if not isinstance(acumulado_list, pd.DataFrame):
                acumulado_list = pd.DataFrame(acumulado_list)

            rain_list.append(acumulado_list)

    return rain_list