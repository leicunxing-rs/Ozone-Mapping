import os
import re
import pyproj
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pyhdf.SD import SD, SDC
import time


def get_ll(path):
    FILE_NAME = path
    DATAFIELD_NAME = 'Optical_Depth_055'
    hdf = SD(FILE_NAME, SDC.READ)

    # Read dataset.
    data3D = hdf.select(DATAFIELD_NAME)
    data = data3D[:, :, :].astype(np.double)

    # 接下来的代码是用于处理经纬度
    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                              (?P<upper_left_x>[+-]?\d+\.\d+)
                              ,
                              (?P<upper_left_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)

    match = ul_regex.search(gridmeta)
    x0 = np.float(match.group('upper_left_x'))
    y0 = np.float(match.group('upper_left_y'))

    lr_regex = re.compile(r'''LowerRightMtrs=\(
                              (?P<lower_right_x>[+-]?\d+\.\d+)
                              ,
                              (?P<lower_right_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = np.float(match.group('lower_right_x'))
    y1 = np.float(match.group('lower_right_y'))

    nx = data.shape[1]
    ny = data.shape[2]
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    xv, yv = np.meshgrid(x, y)

    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("EPSG:4326")
    lon, lat = pyproj.transform(sinu, wgs84, xv, yv)
    return lon, lat


# get the data and qa
def get_data(path):
    hdf = SD(path, SDC.READ)
    data3D = hdf.select('Optical_Depth_055')

    data = data3D[:, :, :].astype(np.double)
    qa = hdf.select('AOD_QA')[:, :, :]
    attrs = data3D.attributes()

    # apply relative attributes of AOD data
    invalid = np.logical_or(data < attrs['valid_range'][0], data > attrs['valid_range'][1])
    invaild = np.logical_or(invalid, data == attrs['_FillValue'])

    # 应用数据挑选指标
    # qa_f = qa.flatten()
    # qa_v = [bin(int(i))[2:].rjust(16, '0')[8:12] != '0000' for i in qa_f]
    # invalid = np.logical_or(invalid, np.array(qa_v).reshape(qa.shape))

    data[invalid] = np.nan
    data = (data - attrs['add_offset']) * attrs['scale_factor']
    return data, qa

# get the data after qa
def get_data_A(path):
    hdf = SD(path, SDC.READ)
    data3D = hdf.select('Optical_Depth_055')

    data = data3D[:, :, :].astype(np.double)
    qa = hdf.select('AOD_QA')[:, :, :]
    attrs = data3D.attributes()

    # apply relative attributes of AOD data
    invalid = np.logical_or(data < attrs['valid_range'][0], data > attrs['valid_range'][1])
    invaild = np.logical_or(invalid, data == attrs['_FillValue'])

    # 应用数据挑选指标
    qa_f = qa.flatten()
    qa_v = [bin(int(i))[2:].rjust(16, '0')[8:12] != '0000' for i in qa_f]  # 仅选取QA为零的数据
    invalid = np.logical_or(invalid, np.array(qa_v).reshape(qa.shape))

    data[invalid] = np.nan
    data = (data - attrs['add_offset']) * attrs['scale_factor']
    return np.nanmean(data, axis = 0)

def multi_year_aod(lon_s, lat_s, hvpath, lon, lat):
    hvtime = time.time()
    D = os.listdir(hvpath)
    D.sort(key=lambda x: int(x[9:16]))
    # 构造一个365*n的数组
    total = np.zeros((len(D), lon_s.shape[0]))
    
    # a和b存储每个站点在此行列号对应的AOD下标
    a = []
    b = []
    for j in range(lon_s.shape[0]):
        a.append([])
        b.append([])
        # 注意，参考论文Li et.al中数据集的分辨率统一为0.1°
        # 实际计算的时候我们生成的分辨率应该为0.01°
        s_o = np.where(np.sqrt(np.square(lon - lon_s[j]) + np.square(lat - lat_s[j])) < 0.1)
        a[j].append(s_o[0])
        b[j].append(s_o[1])
    print('该行列号中的站点有{}个'.format(len(a)))
        
    # 计算多个站点365天的日均值
    for i in range(len(D)):   #len(D)
        time2 = time.time()
        fpath = os.path.join(hvpath, D[i])
        print(fpath)
        data,qa = get_data(fpath)
        
        # 遍历该行列号对应的站点
        for j in range(lon_s.shape[0]):
            day_time = np.zeros(data.shape[0])
            # 遍历该AOD对应的时间波段
            for k in range(data.shape[0]):
                data_s = data[k,:,:]
                qa_s = qa[k,:,:]
                
                # 添加AOD的QA指标属性
                qa_s_around = qa_s[a[j][0],b[j][0]]
                
                #此处采用3个指标
#                 qa_v = [bin(int(t))[2:].rjust(16, '0')[8:12] == '0000' and bin(int(t))[2:].rjust(16, '0')[5:8] == '000' and bin(int(t))[2:].rjust(16, '0')[0:3] == '001' for t in qa_s_around]
                
                # 一般采取QA指标
                qa_v = [bin(int(i))[2:].rjust(16, '0')[8:12] == '0000' for i in qa_s_around]
                # 每个station的AOD：对周围0.1°的AOD取平均
                day_time[k] = np.nanmean(data_s[a[j][0],b[j][0]][qa_v])
            # 将第i天、第j个站点的数据写入
            total[i, j] = np.nanmean(day_time)
        print('time is ',time.time() - time2)
    print('the hv time is', time.time() - hvtime)
    return total

def year_s_region(lon_s, lat_s, hvpath, lon, lat):
    hvtime = time.time()
    D = os.listdir(hvpath)
    D.sort(key=lambda x: int(x[9:16]))
    # 构造一个365*n的数组
    total = np.zeros((len(D), lon_s.shape[0]))
    
    # 以欧式距离0.01为准，找到每个站点对应的AOD数值
    # 其次，使用数组a和b分布存储站点对应AOD数值的经度和纬度
    
    # 第一个版本中的输出为N*K
    # 区域版本的输出为N*7*7*k，即N*49*k
    # 因为AOD数据的经度在同一经度上呈现等距分布，但是其等距随着纬度的降低而降低。
    # 但是MAIAC AOD的数据在同一纬度上等距分布，距离不随着经度的改变而改变。
    # 即存在一种匹配方法，先匹配纬度，再根据同一条纬度上的经度等距，找到合适的经度
    
#     print('先找到每个站点的区域数据所对应的AOD数值索引')
    a, b = [], []
    for j in range(lon_s.shape[0]): # 遍历该行列号里面的每一个数组，找到它们匹配到的站点     # lon_s.shape[0]
        region_cen = lon_s[j], lat_s[j]
        step = 0.01
        region_lon = np.arange(region_cen[0]-step*3, region_cen[0]+step*3, step)
        region_lat = np.arange(region_cen[1]-step*3, region_cen[1]+step*3, step)
        
        a.append([])
        b.append([])
        # 注意，参考论文Li et.al中数据集的分辨率统一为0.1°
        # 实际计算的时候我们生成的分辨率应该为0.01°
        
        # 第一种匹配方法，直接根据欧氏距离进行匹配
    #     s_o = np.where(np.sqrt(np.square(lon - lon_s[j]) + np.square(lat - lat_s[j])) < 0.01)  # 这是第一种匹配方法
    
        # 第二种匹配方法，先匹配纬度，然后匹配经度, 
        # 注意第二种匹配方式，只能得到该站点对应的最近的一个AOD数值
        grid_lat_gap = (lat[:, 0][0] - lat[:, 0][-1])/1199     # 不同经度线上的纬度等距里相等
        for r_lat in region_lat:
            
            # 得到每个格子纬度对应的AOD纬度下标
            if r_lat < lat[:, 0].min():
                print('纬度{:.2f}超出该行列块下侧范围，使用{:.2f}代替！'.format(r_lat, lat[1199, 0]))
                lat_index = np.array([1199])
            elif r_lat > lat[:, 0].max():
                print('纬度{:.2f}超出该行列块上侧范围，使用{:.2f}代替！'.format(r_lat, lat[0, 0]))
                lat_index = np.array([0])
            else:
                lat_index = np.where(abs(r_lat - lat) < grid_lat_gap*0.5)[0]   # 此处的lat_index为一个空数组或者一个常数数组
            
            for r_lon in region_lon:
                
                # 根据纬度确认的经度线，得到经度线上的等距距离
                grid_lon_gap = (lon[lat_index[0],-1] - lon[lat_index[0],0])/1199
                if r_lon > lon[lat_index[0], :].max():
                    print('经度{:.2f}超出该行列号右侧范围，使用{:.2f}代替！'.format(r_lon, lon[lat_index[0], 1199]))
                    lon_index = np.array([1199])
                elif r_lon < lon[lat_index[0], :].min():
                    print('经度{:.2f}超出该行列号左侧范围，使用{:.2f}代替！'.format(r_lon, lon[lat_index[0], 0]))
                    lon_index = np.array([0])
                else:
                    lon_index = np.where(abs(r_lon - lon[lat_index[0], :]) < grid_lon_gap*0.5)[0]   # 此处的lat_index为一个空数组或者一个常数数组            

    
                a[j].append(lon_index[0])
                b[j].append(lat_index[0])
    print('该行列号中的站点有{}个'.format(len(a)))
    a, b = np.array(a), np.array(b)
    # 此刻出现问题:对于某些站点的区域数据可能超出该行列号的范围
        
    
    hvtime = time.time()
    for t in range(len(D)):   # 首先遍历天数，因为AOD的数据以天数为基准进行存储
        daytime = time.time()
        fpath = os.path.join(hvpath, D[t])

        data,qa = get_data(fpath)
        
        s_region_k = np.zeros((lon_s.shape[0], a.shape[1], data.shape[0]))
        for k in range(data.shape[0]):  # 其次遍历时间波段，计划生成站点数*7*7*波段数的数据
            data_k = data[k,:,:]
            qa_k = qa[k,:,:]    # 此处的data_s以及qa_s的大小均为1200*1200
    
            for s in range(lon_s.shape[0]):    #  第三遍历该行列号包含的站点， 计划输出站点数*7*7的数据
    #             day_time = np.zeros(data.shape[0])    
            
                for r in range(a.shape[1]):      # 第四遍历每个站点所包含的区域数据，计划输出
    
                    lon_r, lat_r = a[s][r], b[s][r]
                    qa_r = qa_k[lat_r, lon_r]   # 得到目标位置的QA值
                    
                    qa_r_bool = bin(int(qa_r))[2:].rjust(16, '0')[8:12] == '0000' 
                    
                    # 也可以采用3个指标
                    # qa_v = [bin(int(t))[2:].rjust(16, '0')[8:12] == '0000' and bin(int(t))[2:].rjust(16, '0')[5:8] == '000' and bin(int(t))[2:].rjust(16, '0')[0:3] == '001' for t in qa_s_around]
    
                    if qa_r_bool:
                        s_region_k[s, r, k] = data_k[lat_r, lon_r]
                    else:
                        s_region_k[s, r, k] = np.nan
        s_region = np.nanmean(s_region_k, axis = 2)[np.newaxis,:]   # 输出的值为12*49
        if t == 0:
            year_s_region = s_region
        else:
            year_s_region = np.vstack([year_s_region, s_region])
        if (t+1)%73 == 0:
            print('已经处理{}%'.format((t+1)/73 *20))
#         print('正在处理:{}，时间:{}'.format(fpath, time.time() - daytime))
    #     print('time is ',time.time() - time2)
#     print('the hv time is', time.time() - hvtime)
    return year_s_region

