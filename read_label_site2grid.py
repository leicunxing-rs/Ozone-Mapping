import xarray as xr
import numpy as np
import pandas as pd

# (1)首先我们从整体的标签中加载所需要的年份
def site2ll():

    """
    （1）读取长时间序列的经纬度
    """
    site = pd.read_csv('F:/2_Joint_Estimation/Source Data/Site_label/全国_20140513-20211231.csv', index_col=0)
    site.columns = pd.to_datetime(site.columns, format='%Y%m%d')  # 将数据的表头设置为datetime序列
    year = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    year_site = site[year[5]]  # 获取指定的数据
    year_site = year_site.dropna(axis=0, how='all')  # 删除全空的站点

    # 根据站点的编码找到站点的经纬度
    code2ll = pd.read_excel('F:/2_Joint_Estimation/Source Data/Site_label/_站点列表/站点列表-2020.04.05起.xlsx').loc[:, '监测点编码':'纬度']
    # 这里code2ll中，存在站点编号不同、但是站点名称以及站点经纬度都相同的情况。因此需要将监测点编码输入
    year_site.index = [i[:5] for i in year_site.index.values]   # 对站点索引的异常值进行处理，如'1001A'
    year_site.index.name = 'code'   # 找出重复的站点：因为年限的不同，可能在其他年份没有数据
    year_site = year_site.groupby(by = 'code').mean()   # 这里的合并是np.nanmean()
    # 接着对code2llj进行处理

    # 这里code2ll中，存在站点编号不同、但是站点名称以及站点经纬度都相同的情况。
    # print(np.unique(code2ll.index).shape)  # 查看当前数组非重复的个数
    # np.where(year_site.index.duplicated()) # 找到重复值的索引
    # year_site.loc[year_site.index[[1499, 1500]]]
    # print(code2ll.index.drop_duplicates().shape)  # 查看经过删除之后的形状， 发现与上文的保持一致 这里是1654 （原始1657）
    code2ll = code2ll.drop_duplicates()  # 这里是1652
    code2ll.set_index('监测点编码', inplace = True)

    s2ll_2019 = code2ll.loc[year_site.index].loc[:, '经度':]
    s2ll_2019.columns = ['lon', 'lat']
    return year_site, s2ll_2019
# s2label, s2ll = site2ll()
# s2ll.to_csv('station2grid/site_ll_2019.csv')
# s2label.to_csv('temp_file/site_label_2019.csv')

# (2)将站点对应的经纬度转为1km格子对应的经纬度
# 因为主要的产品分辨率设置不定，因此PM的设置也不定
# 同时调整格子的经纬度精度为1km
def site2g1km():
    """
    (1) 先将站点的经纬度转为格子
    这里的数据，由read_label_year.py文件产生
    """
    s2ll = pd.read_csv('E:/3_Atmos/O3_Mapping/station2grid/site_ll_2019.csv', index_col=0)
    # s2ll = pd.read_csv('E:/3_Atmos/dataset/site_grid_assess/site_ll_2019.csv', index_col =0)    # 使用经过月、年处理的数据
    s2ll.columns = ['lon', 'lat']

    """
    格子的分辨率取1km的时候，这里存在1509个站点， 预计对应1495个格子
    s2ll = s2ll.round(2)   # 经纬度等小数保存的时候最好采用 float64的格式
    调整至其他的分辨率
    """
    def site21km(s2ll, scale=0.01):
        res, quo = np.modf(s2ll*100)
        g2ll = quo.astype('int')/100
        g2ll[res>=0.5]+=scale               # 就处理的细节部分写成函数，作为核心代码避免由于超参数的更改而更改
        print((np.abs(s2ll-g2ll)>0.05).sum())  # 检查
        return g2ll
    g2ll = site21km(s2ll).round(2)


    re_index = np.where(g2ll.duplicated())[0]    # 找到数值索引
    re_s2ll = g2ll.loc[g2ll.index[re_index]]  # 找到编码，即存在一个格子内包含多个站点
    # g2ll = g2ll[~g2ll.duplicated()]         # 在新的分辨率中，合并重复的经纬度，注意在这里无法保存站点的编号信息

    site_label = pd.read_csv('E:/3_Atmos/O3_Mapping/temp_file/site_label_2019.csv', index_col = 0)

    # 找到重复格子对应的code，计算它们的平均值
    g2ll.columns = ['lon', 'lat']
    for g in re_s2ll.values:
        lon_g, lat_g = g[0], g[1]
        index_g = g2ll.index[((g2ll.lon == lon_g) & (g2ll.lat == lat_g))]
        print(index_g)
        site_label.loc[index_g[0]] = site_label.loc[index_g].mean()  # 将重叠的site合并到第一个站点
        site_label = site_label.drop(index_g[1:], axis=0)  # 删除其他的站点
    return site_label, g2ll[~g2ll.duplicated()]

# grid_label, g2ll = site2g1km()
# g2ll.to_csv('E:/3_Atmos/O3_Mapping/station2grid/grid_ll_2019.csv')
# s2ll = pd.read_csv('E:/3_Atmos/O3_Mapping/station2grid/site_ll_2019.csv', index_col=0)
# s2g = s2ll.loc[g2ll.index]
# dif = np.abs(s2g-g2ll)
# grid_label.to_csv('E:/3_Atmos/O3_Mapping/temp_file/grid_label_2019.csv')#

# 将站点调整值其他的分辨率
def site2g5km():
    # s2ll = pd.read_csv('station2grid/site_ll_2019.csv', index_col=0)
    s2ll = pd.read_csv('../../dataset/site_grid_assess/site_ll_2019.csv', index_col=0)
    s2ll.columns = ['lon', 'lat']   # 重新设置pandas的列
    # 格子的分辨率取1km的时候，这里存在1509个站点， 预计对应1495个格子
    # s2ll = s2ll.round(2)
    # 调整至其他的分辨率
    scale = 0.05  # 单位为°
    quo = (s2ll//scale/20).values.reshape(-1)
    res = (s2ll%scale).values.reshape(-1)
    quo[res>scale*0.5] += scale
    s2ll.lon, s2ll.lat = quo.reshape(s2ll.shape)[:, 0], quo.reshape(s2ll.shape)[:,1]   # 得到更改后的站点的经纬度精度

    # 找到处于一个格子里面的站点
    re_index = np.where(s2ll.duplicated())[0]
    re_s2ll = s2ll.loc[s2ll.index[re_index]]  # 找到重复站点对应的经纬度
    g2ll = s2ll[~s2ll.duplicated()]

    # site_label = pd.read_csv('temp_file/site_label_2019.csv', index_col = 0)
    site_label = pd.read_csv('../../dataset/site_grid_assess/site_ozone_2019.csv', index_col=0)
    # 找到重复格子对应的code，计算它们的平均值

    for g in np.unique(re_s2ll, axis= 0):
        lon_g, lat_g = g[0], g[1]
        index_g = s2ll.index[((s2ll.lon == lon_g) & (s2ll.lat == lat_g))]  # 找到该点对应的站点编码
        print(index_g)
        site_label.loc[index_g[0]] = site_label.loc[index_g].mean()  # 将重叠的site合并到第一个站点
        site_label = site_label.drop(index_g[1:], axis=0)  # 删除其他的站点
    return site_label, g2ll
# grid_label, g2ll = site2g5km()
# g2ll.to_csv('E:/3_Atmos/O3_Mapping/station2grid/grid_ll_2019.csv')
# grid_label.to_csv('../../dataset/grid5km_ozone_2019.csv')

