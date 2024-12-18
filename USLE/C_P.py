import rasterio
import pandas as pd
import numpy as np
import gc
from Save_tif import resize
from noise_cleaning import clean_pixel_noise


def calc_C_P(C_P_Table_path: str, C_P_Map_path: str, geo, clean: bool) -> np.ndarray:
    """
    Reclassifies data from a C_P table and a C_P map.

    Parameters
    ----------
    C_P_Table_path : str
        Path to the C_P table Excel file.
    C_P_Map_path : str
        Path to the C_P map raster file.
    clean : bool
        Flag to indicate if the previous C_P Map should be cleaned from pixel noise.
    Returns
    -------
    np.ndarray C_P map array.
    """

    with rasterio.open(C_P_Map_path) as src:
        df_Map_C_P = src.read(1).astype(float)
        if clean:
            df_Map_C_P = clean_pixel_noise(df_Map_C_P, filter_size=4)

    df_Table_cp = pd.read_excel(C_P_Table_path)

    value_to_cp = dict(zip(df_Table_cp.Valor, df_Table_cp.C_P))

    for value, cp_value in value_to_cp.items():
        df_Map_C_P[df_Map_C_P == value] = cp_value

    cp_array = resize(geo, df_Map_C_P)

    return cp_array
