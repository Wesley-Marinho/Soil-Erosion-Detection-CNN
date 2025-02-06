# %%
from USLE.Rain_data import media_rain
from USLE.R import calc_R
from USLE.K import calc_K
from USLE.L_S import calc_L, calc_S
from USLE.C_P import calc_C_P
from USLE.USLE import calc_USLE

from USLE.Save_tif import (
    save_tif,
    resize,
    save_to_netcdf,
)

import rasterio

# %%

C_P_Table_path = "/media/wesley/novo_volume/USLE/C_P/Dados iniciais/C_P_MAPBIOMAS.xlsx"
C_P_Map_path = "/media/wesley/novo_volume/USLE/C_P/Dados iniciais/C_P_MAPBIOMAS_30M.tif"

with rasterio.open(C_P_Map_path) as src:
    geo = src.read(1)

C_P = calc_C_P(C_P_Table_path, C_P_Map_path, geo, clean=False)

C_P_path = "/media/wesley/novo_volume/USLE/C_P/C_P_prev.tif"

save_tif(C_P_path, C_P_Map_path, C_P)

del C_P
# %%

mde_file = "/media/wesley/novo_volume/USLE/L_S/Dados iniciais/FABDEM_MG.tif"
L_path = "/media/wesley/novo_volume/USLE/L_S/L.tif"
S_path = "/media/wesley/novo_volume/USLE/L_S/S.tif"
geo_path = "/media/wesley/novo_volume/USLE/L_S/Dados iniciais/FABDEM_MG.tif"

L = calc_L(mde_file)
save_tif(L_path, geo_path, L)
del L

S = calc_S(mde_file)
save_tif(S_path, geo_path, S)
del S
# %%

data_list = [
    "/media/wesley/novo_volume/USLE/R/Dados iniciais/pr_Tmax_Tmin_NetCDF_Files/pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.2.nc",
    "/media/wesley/novo_volume/USLE/R/Dados iniciais/pr_Tmax_Tmin_NetCDF_Files/pr_20010101_20200731_BR-DWGD_UFES_UTEXAS_v_3.2.2.nc",
    "/media/wesley/novo_volume/USLE/R/Dados iniciais/pr_Tmax_Tmin_NetCDF_Files/pr_20200801_20221231_BR-DWGD_UFES_UTEXAS_v_3.2.2.nc",
]

lat_lon_list = "/media/wesley/novo_volume/USLE/R/Dados iniciais/pr_Tmax_Tmin_NetCDF_Files/lat_lon.csv"

final = media_rain(data_list, lat_lon_list)

# 90M
n_raster_cells_x = 13426
n_raster_cells_y = 10426

geo_path = "/media/wesley/novo_volume/USLE/L_S/Dados iniciais/FABDEM_MG.tif"

R = calc_R(final, n_raster_cells_x, n_raster_cells_y)

R_paths = [
    "/media/wesley/novo_volume/USLE/R/R.tif",
    "/media/wesley/novo_volume/USLE/R/R_95_maior.tif",
    "/media/wesley/novo_volume/USLE/R/R_95_menor.tif",
]

R_r = []

with rasterio.open(geo_path) as src:
    geo = src.read(1)

for item in R:
    R_r.append(resize(geo, item))

del geo, R

for R_df, path in zip(R_r, R_paths):
    save_tif(path, geo_path, R_df)

del R_r
# %%

K_Table_path = "/media/wesley/novo_volume/USLE/K/Dados iniciais/fator_K_simples.xlsx"
K_Map_path = "/media/wesley/novo_volume/USLE/K/Dados iniciais/Solo_MG.shp"
K_path = "/media/wesley/novo_volume/USLE/K/K.tif"
geo_path = "/media/wesley/novo_volume/USLE/L_S/Dados iniciais/FABDEM_MG.tif"

K = calc_K(K_Map_path, K_Table_path, geo_path, K_path)

# %%

R_files = [
    "/media/wesley/novo_volume/USLE/R/R.tif",
    "/media/wesley/novo_volume/USLE/R/R_95_maior.tif",
    "/media/wesley/novo_volume/USLE/R/R_95_menor.tif",
]

S_file = "/media/wesley/novo_volume/USLE/L_S/S.tif"

L_file = "/media/wesley/novo_volume/USLE/L_S/L.tif"

K_file = "/media/wesley/novo_volume/USLE/K/K.tif"

C_P_file = "/media/wesley/novo_volume/USLE/C_P/C_P.tif"

geo_path = "/media/wesley/novo_volume/USLE/L_S/Dados iniciais/FABDEM_MG.tif"

print("Chegou em calc USLE")

shp_path = (
    "/media/wesley/novo_volume/USLE/Arquivos de recorte/limites - MG/MG_limites.shp"
)

A_Paths_tiff = [
    "/media/wesley/novo_volume/USLE/USLE/A.nc",
    "/media/wesley/novo_volume/USLE/USLE/A_95_maior.nc",
    "/media/wesley/novo_volume/USLE/USLE/A_95_menor.nc",
]

A = calc_USLE(S_file, L_file, R_files, K_file, C_P_file)

color_map = {
    1: [64, 64, 64],
    2: [236, 236, 236],
    3: [252, 230, 220],
    4: [246, 178, 148],
    5: [226, 94, 88],
    6: [202, 1, 32],
}

for a, A_Path in zip(A, A_Paths_tiff):
    save_to_netcdf(A, A_Path, color_map)
