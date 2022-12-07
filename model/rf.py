import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import xarray as xr
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

def rf(dataset, retrain = False, modelpath = r'E:\3_Atmos\single_pixel_model\test'):
    # generate the save_model
    train_features, test_features, train_label, test_label = train_test_split(dataset[:, :-1], dataset[:, -1].ravel(),
                                                                              test_size=0.2, random_state=42)
    feature_len = dataset.shape[1] - 1  # 注意特征的个数

    if retrain:
        print('train start!', time.strftime('%H:%M:%S', time.localtime()))
        rf = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=42)
        rf.fit(train_features, train_label)  # 此处将train_labels改为一维向量则warning.   # torch.squeeze(train_labels)
        print('train over')

        joblib.dump(rf, modelpath+'.pkl')  # 保存时，注意当前特征的个数
    else:
        print('start load save_model!', time.strftime('%H:%M:%S', time.localtime()))
        rf = joblib.load(modelpath+'.pkl')


    train_pred = rf.predict(train_features)
    test_pred = rf.predict(test_features)

    print('the train result and test result:', r2_score(train_label, train_pred),
          r2_score(test_label, test_pred))
    print('end!', time.strftime('%H:%M:%S', time.localtime()))


# (1) load label  # 这里分段处理是jupyter的特性
# 合并处于同一格子里面的站点
def load_label():
    # site_label = pd.read_csv('../Site_label/站点_20150101-20151231.csv', index_col = 0).values  # 站点数*天数
    # s2g = pd.read_csv('../Site_label/site2grid_index_2015.csv', index_col = 0).values  # 这里是因为气象数据对应的格子里会有多个地基站点
    # for i in range(s2g.shape[0]):
    #     grid_data = np.mean(site_label[s2g[i][~np.isnan(s2g[i])].astype(int), :], axis = 0)  # 将位于同一格子的站点合并， 得到其全年的值
    #     if i == 0:
    #         grid_label = grid_data.reshape(1, -1)
    #     else:
    #         grid_label = np.concatenate([grid_label, grid_data.reshape(1, -1)], axis = 0)
    # np.save('temp_file/grid_label_2015.npy', grid_label)
    # grid_label = np.load('temp_file/grid_label_2015.npy')  # 这里的形状1477*365
    grid_label = pd.read_csv('temp_file/grid_label_2019.csv', index_col = 0).values  # 这里的形状 1509*365
    # grid_label = pd.read_csv('temp_file/grid_label_2019_5km.csv', index_col = 0).values  # 这里的形状 1509*365
    return grid_label

# (2) load the features
# 2.1 load the new features,
def load_feature():
    # single = xr.open_dataset('temp_file/grid_single_R13_region_2015.nc')
    single = xr.open_dataset('E:/3_Atmos/O3_Mapping/temp_file/grid_single_e13_region_2019.nc')
    # pressure = xr.open_dataset('E:/3_Atmos/O3_Mapping/temp_file/grid_pressure_2_center_2015.nc')
    pressure = xr.open_dataset('E:/3_Atmos/O3_Mapping/temp_file/grid_pressure_2_center_2019.nc')
    # ndvi = pd.read_csv('../ERA5/single/numpy_2015/NDVI_1km.csv', index_col = 0)  # 这里的大小是365*1477
    single_var = ['u10', 'v10', 't2m', 'blh', 'e', 'sp', 'tp', 'uvb', 'ssr', 'str', 'tco3']  # u10和v10的两个替换：ws, wd 'ws', 'wd',
    for i in range(len(single_var)):
        temp = single[single_var[i]].data[:, :, 3, 3][:, :, np.newaxis]
        if i == 0:
            f_single = temp
        else:
            f_single = np.concatenate([f_single, temp], axis = 2)
    pressure_var = ['r', 'o3']   # 这里的r和O3代表RH和Ozone
    for i in range(len(pressure_var)):
        temp2 = pressure[pressure_var[i]].data[:, :, np.newaxis]  # 暂时使用中心数据
        if i == 0:
            f_pressure = temp2
        else:
            f_pressure = np.concatenate([f_pressure, temp2], axis = 2)# f2 = np.concatenate([f,pressure], axis = 2)
    # no2 = np.load('temp_file/grid_month_no2_201901_5km.npy')[:, :, 3, 3][:, :, np.newaxis]
    no2 = np.load('temp_file/grid_year_no2_2019.npy')[:, :, 3, 3][:, :, np.newaxis]
    hcho = np.load('temp_file/grid_year_HCHO_2019.npy')[:, :, 3, 3][:, :, np.newaxis]

    meic = np.load('temp_file/grid_day_region_meic_2019.npy')[:, :, :, 3, 3].swapaxes(0, 1).swapaxes(1, 2)  # 这里的形状4*1495*365统一格式
    st = np.load('temp_file/grid_year_st_2019.npy')

    # (3)  generate the dataset
    features = np.concatenate([f_single, f_pressure, no2, hcho, meic, st], axis = 2)      # 这里的形状 1477*365*k
    # features = np.concatenate([no2[:, :1]], axis = 2)      # 这里的形状 1477*365*k
    return features

def model():
    grid_label = load_label()   # 1495*365
    features = load_feature()   # 1495*365*23
    dataset = np.concatenate([features.reshape(-1, features.shape[2]), grid_label.reshape(-1, 1)], axis = 1)  # [:, :31]

    # 注意，不同的特征个数对应不同的数据集大小，也可能对应不同的性能
    # 分析不同的输入对结果的影响
    subds = np.hstack([np.arange(13), np.arange(13, 15), np.arange(19, 24)])
    # subds = np.arange(24)
    dataset = dataset[:, subds]  # 取一部分特征取数据集，并剔空数据集和建模

    # 分析每个特征的损失率
    # Downward UV radiation at the surface(uvb), surface net solar radiation(ssr)
    # surface net thermal radiation(str), total column ozon(tco3)
    varname = ['u10', 'v10', 't2m', 'blh', 'e', 'sp', 'tp', 'uvb', 'ssr', 'str', 'tco3',    # 0-10
               'rh', 'or',        # 11-12
               'no2','hcho',      # 13-14   (存在缺失数据)
               'industry', 'power', 'residential', 'transportation',   # 15-18 （存在缺失数据）
               'lon', 'lat', 'day', 'month',             # 19-22
               'Ozone'
               ]
    print(np.array(varname)[subds], (~np.isnan(dataset[:, -1])).sum()/len(dataset)) # 统计有效标签的比例
    dataset = dataset[~np.isnan(dataset).any(axis=1)]  # 删除空值, 有效的样本数：18%

    print('the used dataset shape is:',dataset.shape)

    # rf(dataset, retrain = False)  # N*24
# save_model()

def cal_vif(dataset, varname):
    # for i in range(23):
    #     print('特征', varname[i], '估算Ozone的效果是')
    #     rf(dataset[:, [i, -1]])

    # 计算剁成共线性关系，VIF， 对于前面的23个推广
    # vif = [variance_inflation_factor(dataset[:, :-1], i) for i in np.arange(23)]
    # vif = np.around(vif, decimals=3)
    # vif_ds = pd.DataFrame(vif.reshape(1, -1), columns = varname, index = ['VIF'])

    # 我们将时空位置排除之后，再次计算方差膨胀因子VIF
    vif = [variance_inflation_factor(dataset[:, :-5], i) for i in np.arange(19)]
    vif = np.around(vif, decimals=3)
    vif_ds = pd.DataFrame(vif.reshape(1, -1), columns = varname[:-5], index = ['VIF'])

    #
    # # the old input
    # m8 = np.load('../ERA5/single/2015/station_meteo_m8.npy')   # 这里的大小是1477*365*8， 这里和小服务器7*7的保持一致
    # ndvi = pd.read_csv('../ERA5/single/2015/NDVI_1km.csv', index_col = 0)  # 这里的大小是365*1477
    #
    # features = np.concatenate([m8, np.expand_dims(ndvi.T, 2)], axis = 2)
    # dataset = np.concatenate([m8.reshape(-1, m8.shape[2]), grid_label.reshape(-1, 1)], axis = 1)
    # dataset = dataset[~np.isnan(dataset).any(axis=1)]


    # 李同文新论文里面的实验设置，