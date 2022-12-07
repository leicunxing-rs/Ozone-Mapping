import os
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
"""
note: 
这里的AOD数据来源于GEE，其中的空值使用零值填充；
scale_factor的值为0.001
经纬上的resolution是0.00898315

ds[ds==0] = np.nan  # 如果是np.nan，那需要先转为float32

"""
aod_path = r'F:\2_Joint_Estimation\Source Data\GEE_AOD_2019_Year\merged.tif'
ds = rxr.open_rasterio(aod_path)
ds.attrs['scale_factor'] = 0.001
ds.attrs['resolution'] = 0.00898315   # 修改后文件较大无法保存
print((ds[0][0]!=0).sum())

# ds0 = ds[0]
# ds0 = np.where(ds0, ds0, np.nan)  # 插值
ll = pd.read_csv(r'E:\3_Atmos\dataset\grid_ll_2019.csv', index_col = 0).values
glon, glat = ll[:, 0], ll[:, 1]
wlon, wlat = np.load(r'E:\3_Atmos\dataset\win_lon.npy'),\
             np.load(r'E:\3_Atmos\dataset\win_lat.npy')
glon, glat = xr.DataArray(glon, dims='point').astype('float32'), xr.DataArray(glat, dims = 'point').astype('float32')

"""
xarray的插值方式：https://www.jianshu.com/p/e884d177ee62

(1) 基于站点的nc，site匹配
存在 ds.sel 和 ds.interp 两种方式，得到站点对应的值
匹配的时候，坐标的精度需要为float32 
(1.1)为了方便匹配，将resolution从0.00898315调整为0.01, 再进行索引匹配
ds0 = ds0.interp(x = wlon, y = wlat, method='linear', kwargs={'fill_value': 'extrapolate'})
site_value = ds0.sel(x = glon, y = glat)

(1.2) 不经过插值，直接进行插值匹配
注意，全年的需要内存15G，单独分为60合并，需要7分钟
"""
def get_site_aod(ds):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    year_site = []
    year_site.append(ds[:60].interp(x = glon, y = glat))
    year_site.append(ds[60:120].interp(x = glon, y = glat))
    year_site.append(ds[120:180].interp(x = glon, y = glat))
    year_site.append(ds[180:240].interp(x = glon, y = glat))
    year_site.append(ds[240:300].interp(x = glon, y = glat))
    year_site.append(ds[300:365].interp(x = glon, y = glat))
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    site_year_aod = xr.concat(year_site, dim = 'band')

    t = np.where(site_year_aod, site_year_aod, np.nan)   # 将零值转为np.nan
    site_year_aod.data = t.astype('float32')
    site_year_aod['band'] = pd.date_range('2019-01-01', '2019-12-31')
    return site_year_aod
# site_aod_2019 = get_site_aod(ds)
# site_year_aod.to_netcdf(r'E:\3_Atmos\dataset\site_year_aod_2019.nc')
site_aod_2019 = xr.open_dataarray(r'E:\3_Atmos\dataset\site_year_aod_20192.nc')



"""
(2) 基于区域的nc, site匹配
"""


# # 以第一天为例绘图
# t =ds[0]*0.001   # scale = 0.001
# t.plot(cmap="Spectral_r", vmin=0, vmax=1)  # Spectral的互逆配色
# plt.show()
