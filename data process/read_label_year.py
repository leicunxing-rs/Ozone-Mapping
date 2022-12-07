import numpy as np
import pandas as pd
"""
GB 3095-2012，
（1）PM2.5连续监测的要求：
日平均：每日至少20个小时平均浓度；
年平均：每月至少27个日均值（2月25），每年至少324个日均值

（2）O3的浓度要求：（测量范围0.1~500）单位ppb和ug/m3
每8小时：至少6小时浓度；（一级100，二级160）
HJ 654-2013
ppm即：mg/L（毫克/升），ppb即：ug/L(微克/L)
ppm是百万分之一，无量纲量， 10-6, , 若密度为1g/ml, 则1ug/ml
ppb是十亿分之一， 10-9
"""
# HJ 654-2013, O3连续监测的要求
#
# 污染物浓度数据有效性的最低要求：
# 每年至少有324个日平均浓度值。每月至少有27个日平均浓度值（二月25个）。每8小时平均至少有6小时平均浓度值
# 即每年缺失值最多41个，每月缺失值最多4个，

def get_label_ll():

    """
    保存数据的时候，注意保存站点编号；保存经纬度的时候，也注意保存站点编号。
    2019年有1641个编码站点，1628个站点存在经纬度
    """
    site_label = pd.read_csv('F:/2_Joint_Estimation/Source Data/Site_label/站点_O3/站点_20190101-20191231.csv', index_col = 0)  # 这里的形状 1641*365, 524441个有效样本
    site_ll = pd.read_csv('F:/2_Joint_Estimation/Source Data/Site_label/_站点列表/站点列表-2020.01.01起.csv', index_col=0) #.values[:, [0, 3, 4]] # 保存站点编号和经纬度

    site_ll_new = pd.read_csv('F:/2_Joint_Estimation/Source Data/Site_label/_站点列表/站点列表-2022.02.13起.csv', index_col=0)

    # 删除整年站点, 行索引使用loc[name]的方式进行索引
    # (1) 删除全年空白的站点
    site_label_valid = site_label.dropna(how='all')  # 删除整年没有值的站点，得 1510*365个

    # 注意，原数据存在误差，如'3207A'这个站点出现了两次
    # （2）找出数值中出现的重复行
    repeat_row = site_label_valid[site_label_valid.duplicated()]
    site_label_valid = site_label_valid.drop(repeat_row.index)  # 删除重复的行

    # （3）对ll进行同步处理相同的行，注意：1961A和1962A的经纬度相同，2487A和2488A的经纬度也相同（但是对应的数值不同），存在两个3207A
    # 1961A(117.541,36.71)->(117.532, 36.6868), 2487A(111.9975,27.725)->(119.9589, 27.8901)
    """注意2019、2022年相同站点代码对应的站点经纬度不相同"""
    site_ll_valid = site_ll.loc[site_label_valid.index][['经度','纬度']]
    site_2022 = site_ll_new.loc[site_label_valid.index][['经度','纬度']]
    site_ll_valid.loc['1961A'] = site_2022.loc['1961A']
    site_ll_valid.loc['2487A'] = site_2022.loc['2487A']  # 对重复（存误）的地方进行处理
    site_ll_valid = site_ll_valid.drop_duplicates(['经度', '纬度'], keep='first')  # drop(1500) # 删除第二次出现
    """经过第二次处理，有效的站点经纬度有1509对"""

    site_label_valid.columns = pd.date_range('2019-01-01', '2019-12-31')

    # (4) 月判断，label.loc['1001A'][label.loc['1001A'].index.month == 1].shape
    """月评价、年平均的要求：每月不少于27天（2月不少于25天），一年不少于324天"""
    """但是我们构造数据集的时候，暂时忽略对于月评价、年评价的指标"""
    def longtime_assessment(site_label_valid, site_ll_valid):
        # 直接使用年缺失值不少于41个进行判断, label[np.isnan(label).sum(axis = 1)<42]
        # 使用月缺失值小于4进行处理
        # 处理之后的有效站点数为：1341
        for index in site_label_valid.index:
            site_year = site_label_valid.loc[index]
            for month in range(1, 13):
                if np.isnan(site_year[site_year.index.month == month]).sum() > 4:  # 最低有效性：每月至少27天
                    print(index, month, np.isnan(site_year[site_year.index.month == month]).sum())
                    site_year[site_year.index.month == month] = np.nan
            if np.isnan(site_year).sum() > 41:
                print(index, np.isnan(site_year).sum())
                site_year[:] = np.nan
        site_label_valid = site_label_valid.dropna(how='all')  # 去除不满足要求的站点
        site_ll_valid = site_ll_valid.loc[site_label_valid.index]
        # 1341*365，共有 476241个样本
    return site_label_valid.astype('float32'), site_ll_valid.astype('float32')

# label, ll = get_label_ll()
# label = pd.read_csv('E:/3_Atmos/dataset/site_grid_assess/site_ozone_2019.csv', index_col = 0)
# ll = pd.read_csv('E:/3_Atmos/dataset/site_grid_assess/site_ll_2019.csv', index_col = 0)  # 1341*2

"""
（2）继续处理PM数据
"""
ll = pd.read_csv('E:/3_Atmos/dataset/grid_ll_2019.csv', index_col = 0)
site_pm = pd.read_csv('E:/3_Atmos/dataset/site_pm_2019.csv', index_col = 0)
grid_pm = site_pm.loc[ll.index]
