import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize


def calc_K(K_Map_path, K_Table_path, geo_path, K_path):
    """
    Reclassifies data from a K table and a K map.

    Parameters:
    K_Map_path (str): Path to the shapefile containing the K factor map.
    K_Table_path (str): Path to the Excel table containing the K factor values.
    geo_path (str): Path to the DEM GeoTIFF file.
    K_path (str): Path to the output K GeoTIFF file.

    Returns:
    None

    """

    shapefile_components = {
        "shp": K_Map_path,
        "shx": K_Map_path.replace(".shp", ".shx"),
        "dbf": K_Map_path.replace(".shp", ".dbf"),
        "cpq": K_Map_path.replace(".shp", ".cpq"),
        "prj": K_Map_path.replace(".shp", ".prj"),
        "qmd": K_Map_path.replace(".shp", ".qmd"),
    }

    df_K = gpd.read_file(shapefile_components["shp"])
    df_K_Table = pd.read_excel(K_Table_path)

    df_K_Table["nome"] = df_K_Table["nome"].str.strip().str.lower()
    df_K["legenda_2"] = df_K["legenda_2"].str.strip().str.lower()

    fator_k_mapping = dict(zip(df_K_Table["nome"], df_K_Table["Fator_K"]))

    df_K["K"] = df_K["legenda_2"].map(fator_k_mapping)

    gdf_K = gpd.GeoDataFrame(df_K["K"], geometry=df_K["geometry"])

    with rasterio.open(geo_path) as src:
        dem = src.read(1)
        transform = src.transform

    with rasterio.open(
        K_path,
        "w",
        driver="GTiff",
        width=dem.shape[1],
        height=dem.shape[0],
        count=1,
        dtype=rasterio.float32,
        crs=gdf_K.crs,
        transform=transform,
    ) as dst:
        shapes = ((geom, value) for geom, value in zip(gdf_K.geometry, gdf_K["K"]))

    with rasterio.open(
        K_path,
        "w",
        driver="GTiff",
        width=dem.shape[1],
        height=dem.shape[0],
        count=1,
        dtype=rasterio.float32,
        crs=gdf_K.crs,
        transform=transform,
    ) as dst:
        shapes = ((geom, value) for geom, value in zip(gdf_K.geometry, gdf_K["K"]))
        burned = rasterize(shapes, out_shape=dem.shape, transform=transform, fill=0)
        dst.write(burned, 1)
