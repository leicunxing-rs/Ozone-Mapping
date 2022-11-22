"""
在与GEE的对比中，暂时不考虑QA
"""
import os
import re
import pyproj
from datetime import datetime
import sys
import xarray as xr

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pyhdf.SD import SD, SDC
import time
from function import get_ll, get_data, get_data_A  # 同级目录的调用
from tqdm import tqdm


def date2day(date):
    """
    :param date: 列表, 以'%Y%m%d'形式的
    :return:
    """
    # d = datetime.strptime('20190101', '%Y%m%d')
    # s = d.timetuple()  # 含有 tm_year, tm_month, tm_mday, tm_wday, tm_yday
    if isinstance(date, list):
        return [datetime.strptime(d, '%Y%m%d').timetuple().tm_yday for d in date]
    if isinstance(date, str):
        return datetime.strptime(date, '%Y%m%d').timetuple().tm_yday


# def day2hvfile(day):
#     """
#     用于分析某天的NASA_AOD和GEE_AOD是否一致
#     :param day: 一年中的某一天
#     :return: 该天对应的22个行列号路径
#     """
#     f = []
#     for i in range(len(Dhv)):
#         file = os.listdir(Dhv[i])
#         file.sort(key=lambda x: int(x.split('.')[1][-3:]))  # 如 'MCD19A2.A2019365.h23v04.006.2020002225542.hdf'
#         hvfile = [os.path.join(Dhv[i], f) for f in file]
#         f.append(hvfile[day - 1])
#     return f


def sinu2ll(sinx, siny):
    """
    :param sinu: 经度和纬度的sinu投影
    :return: 经度和纬度坐标
    """
    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    xv, yv = np.meshgrid(sinx, siny)
    lon, lat = pyproj.transform(sinu, wgs84, xv, yv)
    return (lon.astype('float32'), lat.astype('float32'))

    # aod = gdal.Open(ds_meta['SUBDATASET_2_DESC'])


# Dir = 'F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019'
# hvs = os.listdir(Dir)
# hvs.sort(key=lambda x: int(x[1:3] + x[4:]))  # 如 h23v04，22个
# Dhv = [os.path.join(Dir, hv) for hv in hvs]


# tar_day = date2day(['20190101', '20190301', '20190601', '20191201'])  # get the tm_yday of the date
# fp = day2hvfile(tar_day[0])   # get the 22 hv filepath of the day
#
# 将单个hdf文件保存为nc
# sinu, ll = get_ll(fp[0])  # 返回sinu投影（2个1200）
# # ori_data = np.nanmean(get_data(fp[0])[0], axis = 0)
# data_qa = get_data_A(fp[0])
# ds = xr.DataArray(data_qa[np.newaxis, :, :].astype('float32'), coords=[[datetime.strptime(fp[0].split('.')[1][1:], '%Y%j')], sinu[1], sinu[0]],
#                   dims=['time', 'lat', 'lon'])

def hdf2nc(inDir, outDir, hvname):
    """
    将输入文件夹中的某hv中的hdf全部转为nc文件
    :param inDir: F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019
    :param outDir: F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019_nc
    :param hvname: h23v04
    :return:
    """
    print(hvname)
    indir, outdir = os.path.join(inDir, hvname), os.path.join(outDir, hvname)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    infiles = os.listdir(indir)
    infiles.sort(key=lambda x: int(x.split('.')[1][-3:]))  # 按照日期排序
    for file in tqdm(infiles):
        sinu, _ = get_ll(os.path.join(indir, file))
        data_qa = get_data_A(os.path.join(indir, file))
        ds = xr.DataArray(data_qa[np.newaxis, :, :].astype('float32'),
                          coords=[[datetime.strptime(file.split('.')[1][1:], '%Y%j')], sinu[1], sinu[0]],
                          dims=['time', 'lat', 'lon'])
        ds.to_netcdf(os.path.join(outdir, hvname + '_' + file.split('.')[1][1:] + '.nc'))
        # if file.split('.')[1][1:] == '2019005':
        #     break


def bool_hdf2nc():
    file1 = 'F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019\\h23v04\\MCD19A2.A2019005.h23v04.006.2019009064401.hdf'
    file2 = 'F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019_nc\\h23v04\\h23v04_2019005.nc'
    sinu_, ll = get_ll(file1)
    ds1 = get_data_A(file1)
    ds2 = xr.open_Dataarray(file2)


def hdfs2nc():
    """
    每个HV块转nc，需要20min-35min
    :return:
    """
    inDir = 'F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019'
    outDir = 'F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019_nc'

    tar_hv = ['h25v03', 'h26v03', 'h24v04', 'h25v04', 'h26v04', 'h27v04']
    for hv in tar_hv:
        hdf2nc(inDir=inDir, outDir=outDir, hvname=hv)

# 设置目标经纬度
tar_lon, tar_lat = np.arange(72.5, 136.26, 0.01), np.arange(54.5, 17.99, -0.01)[:-1]

def nc2stitch(Dir='F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019_nc', dayindex=0):
    """
    输入为保存有nc文件的hv子文件夹的目录，day为需要处理的天索引
    将每天的nc_hv处理为我们的数据，需要7min左右
    :return:
    """

    i = dayindex  # 取第一天
    files = [os.listdir(os.path.join(Dir, hv))[i] for hv in hv_nc]  # 找到某天对应的22个行列nc文件
    v_tiles = ['03', '04', '05', '06', '07']  # 即5行, （2,5,6,5,2）

    # （1）将22个nc文件，每行先合并，然后在根据维度拼接
    ds_h = []
    for v_tile in v_tiles:
        files_v = [f for f in files if f.split('_')[0][-2:] == v_tile]  # 找到某一行的nc
        ds_v = xr.concat([xr.open_dataarray(os.path.join(outDir, f.split('_')[0], f)) for f in files_v], dim='lon')
        ds_h.append(ds_v)
    ds_tile = xr.concat(ds_h, dim='lat')
    # ll_tile = sinu2ll(ds_tile.lon.data, ds_tile.lat.data) # 获取坐标

    #（2） 拼接好的nc文件，裁剪并插值为我们需要的范围
    line_interp = []
    for i in tqdm(range(len(ds_tile.lat))):  # 对维度对应的每一行，依次处理
        ds_line = ds_tile[:, i:i+1, :]

        ll_line = sinu2ll(ds_line.lon.data, ds_line.lat.data)   # sinu2ll
        ll_line = [ll[0] for ll in ll_line]
        ll_line[0] = (ll_line[0]+360)%360  # 处理负数的lon

        ds_line.coords['lon'], ds_line.coords['lat'] = ll_line[0], ll_line[1][:1] # reset lon and lat
        ds_interp = ds_line.interp(lon=tar_lon).astype('float32')  # 将原始行的数据，插值到目标的位置
        line_interp.append(ds_interp)

    line_com = xr.concat(line_interp, dim='lat').astype('float32') # 按维度合并
    interp_com = line_com.interp(lat = tar_lat).astype('float32')
    return interp_com

outDir = 'F:\\2_Joint_Estimation\\Source Data\\NASA_AOD2019_nc'
hv_nc = os.listdir(outDir)
hv_nc.sort(key=lambda x: int(x[1:3] + x[-1:]))  # 'h23v04'
day_nc = nc2stitch(Dir=outDir, dayindex=date2day('20191201')-1)
# day_nc.to_netcdf('nasa-aod_20191201.nc')


def plot(dss):
    """
    遥感反演的气溶胶光学厚度本质上是整个大气的消光系数，可以近似地理解为与大气顶层的太阳辐射能相比，地面得到的辐射能占其几分之一；
    由于我们想要的是大气污染程度而不是到底接收到多少能量，所以取了一个接近无污染的时间，把它的AOD设为0，记下这时的大气条件；
    再找这个地方史上污染最严重时刻，把它的AOD设为1，同样记录大气条件。然后把这套标量应用到相关的辐射传输模型里，大家都用，
    这样子就跟其它指数一样可以在0到1之间变动了。
    :return:
    """
    for ds in dss:
        # ds = day_nc
        ds.plot(cmap = "Spectral_r", vmin=0, vmax=1)  # Spectral的互逆配色
        plt.savefig('nasa-aod_'+ds.time.data.astype(str)[0][:10]+'.png')
        plt.show()

"""
行列号的排列顺序：
v03: h25,h26, 2个，
v04: h23~h27, 5个，
v05: h23~h28，6个，
v06: h25~h29，5个，
v07: h28,h29，2个，这里存在海南的下半部分，因此v08部分不考虑
v08: h28,h29，2个, 
"""
