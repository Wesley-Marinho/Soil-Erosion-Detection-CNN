import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import cv2


def calculate_fator_R(df_final: pd.DataFrame) -> list:
    """
    Calculates the Fator R for a given dataframe.

    Parameters
    ----------
    df_final : pd.DataFrame
        The dataframe containing the required data.

    Returns
    -------
    list
        A list containing the Fator R value and the coordinates.

    """
    aux = {}
    aux["data"] = pd.to_datetime(df_final.iloc[:, 0])

    latitude = df_final["LATITUDE"][1]
    longitude = df_final["LONGITUDE"][1]
    media_mensal = df_final.groupby("MES")["chuva"].mean()
    P_anual = media_mensal.sum()

    fator_R = sum(67.355 * ((mes**2 / P_anual) ** 0.85) for mes in media_mensal)

    ponto = [fator_R, float(longitude), float(latitude)]

    return ponto


def calc_El30(rain_list: list) -> list:
    """
    Calculates the El30 index for a list of rainfall dataframes.

    Parameters
    ----------
    rain_list : list
        A list of rainfall dataframes, one for each year.

    Returns
    -------
    El30Anual_list : list
        A list of El30 index dataframes, one for each year. Each dataframe contains the El30 index values, longitude, and latitude for each event.

    """
    El30Anual = []
    longitude = []
    latitude = []

    for df_principal in rain_list:
        ponto = calculate_fator_R(df_principal)

        El30Anual.append(ponto[0])
        longitude.append(ponto[1])
        latitude.append(ponto[2])

    El30Anual_original = El30Anual
    El30Anual_maior = [x * 1.95 for x in El30Anual_original]
    El30Anual_menor = [x * 0.05 for x in El30Anual_original]

    El30Anual_lista = [El30Anual_original, El30Anual_maior, El30Anual_menor]

    El30Anual_list = []
    for El30Anual in El30Anual_lista:
        df = pd.DataFrame(
            {"El30Anual": El30Anual, "Longitude": longitude, "Latitude": latitude}
        )
        El30Anual_list.append(df)

    return El30Anual_list


def df_to_shapefile(dataframe):
    """
    Convert a GeoDataFrame to a Shapefile.

    Parameters
    ----------
    dataframe : GeoDataFrame
        The GeoDataFrame to convert to a Shapefile.

    Returns
    -------
    gdf : GeoDataFrame
        The converted Shapefile as a GeoDataFrame.

    """
    geometry = gpd.points_from_xy(dataframe["Longitude"], dataframe["Latitude"])
    gdf = gpd.GeoDataFrame(dataframe, geometry=geometry)
    return gdf


def extract_coordinates(gdf, n_raster_cells_x, n_raster_cells_y):
    """
    Extract the coordinates from a GeoDataFrame and interpolate them to a raster.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame containing the coordinates.
    n_raster_cells_x : int
        The number of cells in the x-axis of the raster.
    n_raster_cells_y : int
        The number of cells in the y-axis of the raster.

    Returns
    -------
    idw : numpy.ndarray
        The interpolated raster.

    """
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values

    atributo_alvo = "El30Anual"
    values = gdf[atributo_alvo].values

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    x_raster = np.linspace(min_x, max_x, n_raster_cells_x)
    y_raster = np.linspace(min_y, max_y, n_raster_cells_y)
    xx, yy = np.meshgrid(x_raster, y_raster)
    raster_points = np.column_stack((xx.flatten(), yy.flatten()))
    idw = calc_idw(x, y, raster_points, values, n_raster_cells_y, n_raster_cells_x)

    return idw  # Retorna o raster interpolado.


def calc_idw(x, y, raster_points, values, n_raster_cells_y, n_raster_cells_x):
    """
    Calculate the inverse distance weighted (IDW) interpolation of a set of points.

    Parameters
    ----------
    x : numpy.ndarray
        The x-coordinates of the points.
    y : numpy.ndarray
        The y-coordinates of the points.
    raster_points : numpy.ndarray
        The x-y coordinates of the raster points.
    values : numpy.ndarray
        The values of the points.
    n_raster_cells_y : int
        The number of cells in the y-axis of the raster.
    n_raster_cells_x : int
        The number of cells in the x-axis of the raster.

    Returns
    -------
    interpolated_raster : numpy.ndarray
        The interpolated raster.

    """
    tree = cKDTree(np.column_stack((x, y)))
    distances, indices = tree.query(raster_points, k=3)

    inverse_distances = 1.0 / distances
    weights = inverse_distances / np.sum(inverse_distances, axis=1)[:, np.newaxis]

    interpolated_values = np.sum(weights * values[indices], axis=1)

    interpolated_raster = interpolated_values.reshape(
        (n_raster_cells_y, n_raster_cells_x)
    )

    interpolated_raster = cv2.flip(interpolated_raster, 0)

    return interpolated_raster


def calc_R(directory, n_raster_cells_x, n_raster_cells_y):
    """
    Calculates the R raster for a given directory, using n_raster_cells_x and n_raster_cells_y as the number of raster cells in the x and y directions, respectively.

    Parameters
    ----------
    directory : str
        The directory containing the El Niño/La Niña data files.
    n_raster_cells_x : int
        The number of raster cells in the x direction.
    n_raster_cells_y : int
        The number of raster cells in the y direction.

    Returns
    -------
    raster_list : list
        A list of rasters, one for each El30 event. Each raster is a NumPy array with dimensions (n_raster_cells_y, n_raster_cells_x).

    """
    El30Anual_list = calc_El30(directory)
    shape_El30_list = []
    for df in El30Anual_list:
        shape_El30_list.append(df_to_shapefile(df))

    raster_list = []
    for gdf in shape_El30_list:
        R = extract_coordinates(gdf, n_raster_cells_x, n_raster_cells_y)
        raster_list.append(R)
    return raster_list