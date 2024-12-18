from osgeo import gdal, osr
import rasterio
import numpy as np
from scipy.ndimage import zoom
import geopandas as gpd
from rasterio.mask import mask
from PIL import Image


def get_tif_gt(tif_add: str) -> tuple:
    """
    Open a GeoTIFF file and retrieve its geotransform.

    Args:
        tif_add (str): Path to the GeoTIFF file.

    Returns:
        tuple: The geotransform of the GeoTIFF file.

    Raises:
        ValueError: If the input file is not a valid GeoTIFF.

    """
    data = gdal.Open(tif_add)

    if data is None:
        raise ValueError(f"{tif_add} is not a valid GeoTIFF file.")

    gt = data.GetGeoTransform()

    return gt


def create_tif(
    tif_add: str, gt: tuple, tif_matrix: np.ndarray, no_data: float = 0, srs: int = 4674
):
    """
    Create a GeoTIFF file from a NumPy array.

    Args:
        tif_add (str): Path to the output GeoTIFF file.
        gt (tuple): The geotransform of the output GeoTIFF file.
        tif_matrix (np.ndarray): The NumPy array to be saved as a GeoTIFF.
        no_data (float, optional): The no data value of the output GeoTIFF.
            Defaults to 0.
        srs (int, optional): The spatial reference system of the output GeoTIFF.
            Defaults to 4674 (WGS 84 / UTM zone 46N).

    Raises:
        ValueError: If the input GeoTIFF path is not a valid file path.

    """
    Y, X = tif_matrix.shape

    driver = gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(tif_add, X, Y, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(gt)
    outBand = outRaster.GetRasterBand(1)
    outBand.SetNoDataValue(no_data)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(srs)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outBand.WriteArray(tif_matrix)
    outBand.FlushCache()

    outRaster = None
    outBand = None
    outRasterSRS = None
    driver = None


def save_tif(destination: str, geo_path: str, tif_array: np.ndarray):
    """
    Save a NumPy array as a GeoTIFF file.

    Args:
        destination (str): Path to the output GeoTIFF file.
        geo_path (str): Path to the input GeoTIFF file.
        tif_array (np.ndarray): The NumPy array to be saved as a GeoTIFF.

    Raises:
        ValueError: If the input GeoTIFF path is not a valid file path.

    """
    gt = get_tif_gt(geo_path)
    create_tif(destination, gt, tif_array)
    print("Arquivo salvo")


def resize(standard: np.ndarray, origem: np.ndarray) -> np.ndarray:
    """
    Resizes an array to a specified shape while maintaining the aspect ratio.

    Args:
        standard (np.ndarray): The desired output shape.
        origem (np.ndarray): The array to be resized.

    Returns:
        np.ndarray: The resized array.

    Raises:
        ValueError: If the input arrays have incompatible shapes.

    """

    resized = zoom(origem, np.array(standard.shape) / np.array(origem.shape), order=1)

    return resized


def save_nc(array, geo_path, data_path, shp_path, nodata_value=0):

    shapefile = gpd.read_file(shp_path)

    with rasterio.open(geo_path) as src:
        out_image, out_transform = rasterio.mask.mask(
            src, shapefile.geometry, crop=False
        )
        out_meta = src.meta
        crs = src.crs

    mask = out_image[0] != src.nodata
    array_masked = np.where(mask, array, nodata_value)

    color_map = {
        1: [64, 64, 64],
        2: [236, 236, 236],
        3: [252, 230, 220],
        4: [246, 178, 148],
        5: [226, 94, 88],
        6: [202, 1, 32],
    }

    rgba_array = np.zeros((4, array.shape[0], array.shape[1]), dtype=np.uint8)
    unique_values = np.unique(array)
    print("Valores únicos no array original:", unique_values)

    for value in unique_values:
        if value not in color_map:
            print(f"Valor {value} não está no mapa de cores!")

    for value, color in color_map.items():
        mask_value = array_masked == value
        rgba_array[0:3, mask_value] = np.array(color).reshape(3, 1)
        print(
            f"Aplicando cor {color} para valor {value}, pixels afetados: {np.sum(mask_value)}"
        )

    # Certificar-se de que o canal alfa está corretamente definido para pixels dentro do valor 6
    rgba_array[3] = np.where(array_masked != nodata_value, 255, 0)

    out_meta.update(
        {
            "driver": "GTiff",
            "height": rgba_array.shape[1],
            "width": rgba_array.shape[2],
            "count": 4,
            "dtype": "uint8",
            "crs": crs,
            "transform": out_transform,
            "nodata": nodata_value,
            "compress": "deflate",
            "zlevel": 5,
            "interleave": "pixel",
        }
    )

    with rasterio.open(data_path, "w", **out_meta) as dest:
        dest.write(rgba_array)
