import numpy as np
import rasterio
import richdem as rd


def calc_slope(mde_file: str) -> np.ndarray:
    """
    Calculates the slope of the terrain in degrees from a Digital Elevation Model (DEM) file.

    Args:
        mde_file (str): The path to the DEM file.

    Returns:
        np.ndarray: A 2D numpy array representing the slope of the terrain in degrees.

    This function reads the elevation data from the provided DEM file and calculates the slope using the gradient of the elevation data.
    The slope is then converted from radians to degrees and returned as a 2D numpy array.
    """
    with rasterio.open(mde_file) as src:
        dem = src.read(1)
        cell_size = src.transform.a

    dzdx, dzdy = np.gradient(dem, cell_size, cell_size)
    slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_degrees = np.degrees(slope_radians)
    return slope_degrees


def calc_beta(slope: np.ndarray) -> np.ndarray:
    """
    Calculates the beta parameter for the Soil.

    Args:
        slope (np.ndarray): A 2D numpy array representing the slope of the terrain in degrees.

    Returns:
        np.ndarray: A 2D numpy array representing the beta parameter.

    """
    beta = (np.sin(slope * 0.01745) / 0.0896) / (
        3 * (np.sin(slope * 0.01745) ** 0.8) + 0.56
    )
    return beta


def calc_M(beta: float) -> float:
    """
    Calculates the M parameter .

    Args:
        beta (float): The beta parameter.

    Returns:
        float: The M parameter.

    """
    M = beta / (1 + beta)
    return M


def calc_flow_direction(mde_file: str):
    """
    Calculates the flow direction of a Digital Elevation Model (DEM) using the
    Flood Fill algorithm.

    Args:
        mde_file (str): The path to the MDE file.

    Returns:
        np.ndarray: A 2D numpy array representing the flow direction of the DEM.

    """
    with rasterio.open(mde_file) as src:
        dem = src.read(1)

    dx, dy = np.gradient(dem, src.res[0], src.res[1])
    flow_direction = np.arctan2(-dy, dx) * 180 / np.pi
    flow_direction = (flow_direction + 360) % 360
    zero_mask = flow_direction == 0
    flow_direction[zero_mask] = 1
    return flow_direction


def calc_flow_accumulation(dem_path: str):
    """
    Calculates the flow accumulation of a Digital Elevation Model (DEM) using the
    D8 algorithm.

    Args:
        dem_path (str): The path to the DEM file.

    Returns:
        np.ndarray: A 2D numpy array representing the flow accumulation of the DEM.

    """
    dem_flow = rd.LoadGDAL(dem_path, no_data=-9999)
    rd.FillDepressions(dem_flow, epsilon=False, in_place=True)
    flow_accumulation = rd.FlowAccumulation(dem_flow, method="D8")
    return flow_accumulation


def calc_L(mde_file: str):
    """
    Calculates the L parameter for the Soil.

    Args:
        mde_file (str): The path to the MDE file.

    Returns:
        np.ndarray: A 2D numpy array representing the L parameter.

    This function calculates the L parameter for the Soil using the provided MDE file.
    It first calculates the slope, beta, and M parameters, then computes the flow direction and accumulation.
    Finally, it uses these values to calculate the L parameter according to the formula:

    L = (((flow_accumulation + 900) ** (M + 1)) - (flow_accumulation ** (M + 1))) / (
        (flow_direction**M) * (30 ** (M + 2)) * (22.13**M)
    )

    The L parameter is an important factor in understanding soil erosion and sediment transport.
    It is used to estimate the potential for soil erosion and to design appropriate soil conservation measures.
    """
    flow_accumulation = calc_flow_accumulation(mde_file)

    slope = calc_slope(mde_file)

    beta = calc_beta(slope)

    del slope

    M = calc_M(beta)

    del beta

    flow_direction = calc_flow_direction(mde_file)

    L = (((flow_accumulation + 900) ** (M + 1)) - (flow_accumulation ** (M + 1))) / (
        (flow_direction**M) * (30 ** (M + 2)) * (22.13**M)
    )

    return L


def flow_dir_acc(mde_file: str):
    """
    Calculates the L parameter for the Soil.

    Args:
        mde_file (str): The path to the MDE file.

    Returns:
        np.ndarray: A 2D numpy array representing the L parameter.

    This function calculates the L parameter for the Soil using the provided MDE file.
    It first calculates the slope, beta, and M parameters, then computes the flow direction and accumulation.
    Finally, it uses these values to calculate the L parameter according to the formula:

    L = (((flow_accumulation + 900) ** (M + 1)) - (flow_accumulation ** (M + 1))) / (
        (flow_direction**M) * (30 ** (M + 2)) * (22.13**M)
    )

    The L parameter is an important factor in understanding soil erosion and sediment transport.
    It is used to estimate the potential for soil erosion and to design appropriate soil conservation measures.
    """
    flow_direction = calc_flow_direction(mde_file)

    flow_accumulation = calc_flow_accumulation(mde_file)

    return flow_direction, flow_accumulation


def calc_S(mde_file: str) -> np.ndarray:
    """
    Calculates the S parameter for the Soil.

    Args:
        mde_file (str): The path to the MDE file.

    Returns:
        np.ndarray: A 2D numpy array representing the S parameter.

    This function calculates the S parameter for the Soil using the provided MDE file.
    It first calculates the slope, then applies a condition to determine the appropriate formula for computing the S parameter.
    The formula used depends on whether the slope is less than 0.09.
    If the condition is met, the function uses the formula: S = 10.8 * sin(slope) + 0.03.
    Otherwise, it uses the formula: S = 16.8 * sin(slope) - 0.5.

    The S parameter is an important factor in understanding soil erosion and sediment transport.
    It is used to estimate the potential for soil erosion and to design appropriate soil conservation measures.
    """
    slope = calc_slope(mde_file)

    condition = np.tan(slope * 0.01745) < 0.09
    S = np.where(
        condition,
        10.8 * np.sin(slope * 0.01745) + 0.03,
        16.8 * np.sin(slope * 0.01745) - 0.5,
    )

    return S
