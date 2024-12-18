import rasterio
from scipy.ndimage import zoom
import numpy as np
import gc


def reclassify_map(data: np.ndarray) -> np.ndarray:
    """
    Reclassifies a NumPy array into a new array with different classes.

    Parameters
    ----------
    data : np.ndarray
        The input NumPy array to be reclassified.

    Returns
    -------
    np.ndarray
        The reclassified NumPy array.

    """
    data_reclassified = data.copy()

    data_reclassified[data < 0] = -9999

    data_reclassified[(data >= 0) & (data < 1)] = 1

    data_reclassified[(data >= 1) & (data < 10)] = 2

    data_reclassified[(data >= 10) & (data < 20)] = 3

    data_reclassified[(data >= 20) & (data < 50)] = 4

    data_reclassified[(data >= 50) & (data < 200)] = 5

    data_reclassified[data >= 200] = 6

    return data_reclassified


def recize_map(shape, origem):
    """
    Resizes a NumPy array to a specified shape while maintaining the aspect ratio.

    Parameters
    ----------
    shape : tuple
        The desired output shape.
    origem : np.ndarray
        The input NumPy array to be resized.

    Returns
    -------
    np.ndarray
        The resized NumPy array.

    """
    resized = zoom(origem, np.array(shape) / np.array(origem.shape), order=1)
    return resized


def calc_USLE(S_file, L_file, R_files, K_file, C_P_file):
    # Carrega e processa S, L, K, C_P
    with rasterio.open(S_file) as src:
        S = src.read(1)
    print("S carregado")

    with rasterio.open(L_file) as src:
        L = src.read(1)
    print("L carregado")

    with rasterio.open(K_file) as src:
        K = src.read(1)
    K = recize_map(L.shape, K)
    print("K carregado")

    with rasterio.open(C_P_file) as src:
        C_P = src.read(1)
    C_P = recize_map(L.shape, C_P)
    print("C_P carregado")

    semi_A = L * S * K * C_P

    # Deleta variáveis intermediárias e limpa memória
    del L, S, K, C_P
    gc.collect()

    # Processa os arquivos R e realiza os cálculos
    A_reclassificado_list = []

    for R_file in R_files:
        with rasterio.open(R_file) as src:
            R = src.read(1)
        print("R carregado")

        A = R * semi_A
        del R  # Deleta a variável R após uso
        gc.collect()

        A_reclassificado = reclassify_map(A)
        A_reclassificado_list.append(A_reclassificado)

        # Libera memória do cálculo e da reclassificação
        del A, A_reclassificado
        gc.collect()

    # Libera memória do semi_A após todos os cálculos
    del semi_A
    gc.collect()

    print("Passou de todos os cálculos e reclassificação")

    return A_reclassificado_list
