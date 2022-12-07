import numpy as np
import pandas as pd
import xarray as xr
from  tqdm import tqdm
from rf import rf

"""
特定输出：
"""
grid_pm = pd.read_csv(r'E:\3_Atmos\dataset\grid_pm_2019.csv', index_col = 0).values[:, :, np.newaxis]  # 这里的形状 1509*365
grid_o3 = pd.read_csv(r'E:\3_Atmos\dataset\grid_o3_2019.csv', index_col = 0).values[:, :, np.newaxis]  # 这里的形状 1509*365

"""
共享输入：气象数据
"""
def meteo_data():
    single = xr.open_dataset('E:/3_Atmos/O3_Mapping/temp_file/grid_single_e13_region_2019.nc')  # 这里包含wswd/uv等重复变量
    pressure = xr.open_dataset('E:/3_Atmos/O3_Mapping/temp_file/grid_pressure_2_center_2019.nc')
    single_var = ['u10', 'v10', 't2m', 'blh', 'e', 'sp', 'tp', 'uvb', 'ssr', 'str', 'tco3']
    for i in range(len(single_var)):
        temp = single[single_var[i]].data[:, :, 3, 3][:, :, np.newaxis]
        if i == 0:
            f_single = temp
        else:
            f_single = np.concatenate([f_single, temp], axis=2)
    pressure_var = ['r', 'o3']  # 这里的r和O3代表RH和Ozone
    for i in range(len(pressure_var)):
        temp2 = pressure[pressure_var[i]].data[:, :, np.newaxis]  # 暂时使用中心数据
        if i == 0:
            f_pressure = temp2
        else:
            f_pressure = np.concatenate([f_pressure, temp2], axis=2)  # f2 = np.concatenate([f,pressure], axis = 2)
    return f_single, f_pressure

sin, pre = meteo_data()

"""
特定输入：卫星数据
"""
no2 = np.load('temp_file/grid_year_no2_2019.npy')[:, :, 3, 3][:, :, np.newaxis]
hcho = np.load('temp_file/grid_year_HCHO_2019.npy')[:, :, 3, 3][:, :, np.newaxis]
aod = xr.open_dataarray(r'E:\3_Atmos\dataset\grid_year_aod_2019.nc').data.T[:, :, np.newaxis]
st = np.load('temp_file/grid_year_st_2019.npy')

"""
构建分支数据集、联合估算数据集
"""
ds_o3 = np.concatenate([sin, pre, no2, hcho, st, grid_pm], axis = 2).reshape(-1, 16)  # 16 /20
ds_o3 = ds_o3[~np.isnan(ds_o3).any(axis=1)]
ds_pm = np.concatenate([sin, pre, aod, st, grid_pm], axis = 2).reshape(-1, 15)
ds_pm = ds_pm[~np.isnan(ds_pm).any(axis=1)]

"""
使用RF模型进行初步估计
o3估算约4分钟，pm估约3分钟
无ST，0.823， 0.772
"""
rf(ds_o3, retrain=True, modelpath=r'E:\3_Atmos\single_pixel_model\o3')
rf(ds_pm, retrain=True, modelpath=r'E:\3_Atmos\single_pixel_model\pm')