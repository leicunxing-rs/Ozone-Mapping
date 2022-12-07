# coding=utf-8
# git 保存至自己的库中
# it's just a tool!
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
data source：include CO2,NO*, O3, https://quotsoft.net/air
站点测量值中，除了CO，其余的单位均为 ug/m3,毫克/立方米
注意单位之间的换算：1ppbv=4.46*10e-8(mol/m3), 1 ug/m3 = 0.467ppbv， 即1ppbv 约等于 2.14ug/m3(韦晶)

日评价，日最大8小时平均：当日8 时至24 时至少有14 个有效8小时平均浓度值。
当不满足14 个有效数据时，若日最大8 小时平均浓度超过浓度限值标准时，统计结果仍有效。
年评价，日最大8小时平均值的第90百分数：日历年内O3 日最大8 小时平均的特定百分位数的有效性规定为日历年内至少有324个O3 日最大8 小时平均值，
每月至少有27 个O3 日最大8 小时平均值（2 月至少25 个O3日最大8 小时平均值）。
'''

# 这里的[O3, O3_24h, O3_8h, O3_8h_24h]分别指O3实时浓度、O3 24小时最大值、
# O3 8小时滑动均值、O3 8小时滑动均值 24h

def dealDayO3(fpath=r'F:\2_Joint_Estimation\Source Data\Site_label\站点_20190101-20191231\china_sites_20190101.csv'):
    """
    :param fpath: 文件路径
    :return: 生成各站点每日对应的最大8小时O3数值（注意，这里不是日均值数据，因此对应的年数据可能也没有相应的均值数据）
    """
    """
    臭氧的评价指标：日最大8小时平均
    对于8小时滑动平均，每8小时最少6小时
    注意：由于文件给出了O3_8h_24h，即O3 8小时滑动均值 的24小时最大值	(微克/立方米)
    """

    print(fpath)
    ds = pd.read_csv(fpath, index_col=0)  # index_col = 0 意思是将第一列设置为索引值
    if ds.empty:
        return ds
    # 关于O3,它是取滑动平均小时最大的哪一个
    # ds_O3 = ds[ds['type'] == 'O3'].iloc[:, 2:]
    # ds_O3_8 = ds[ds['type'] == 'O3_8h'].iloc[:, 2:] #
    ds_O3_8_24 = ds[ds['type'] == 'O3_8h_24h'].iloc[:, 2:]  # 'O3_8h_24h'由原网站给出
    return ds_O3_8_24.iloc[ds_O3_8_24.shape[0] - 1, :]  # 返回各个站点每天最大8小时平均的值, 也就是该天最后一个小时的O3 8小时滑动均值的24小时最大值

def dealDayPM(fpath=r'F:\2_Joint_Estimation\Source Data\Site_label\站点_20190101-20191231\china_sites_20190101.csv'):
    """
    :param fpath: 文件路径
    :return: 返回每天的PM2.5均值
    """
    """
    注意，文件中提供了'PM2.5_24h'，可以供我们使用。如第一天，根据PM_24指标发现第一天缺失值的站点数为180, 根据小时值的计算结果为162个
    out = day_pm.iloc[-1]       # 使用最后一小时的24小时滑动平均计算
    hour_pm = ds[ds['type'] == 'PM2.5'].iloc[:, 2:]
    out2 = np.nanmean(hour_pm, axis=0)  # 根据每小时均值计算，需要注意，每天至少有18个小时值
    """
    print(fpath)
    ds = pd.read_csv(fpath, index_col=0)  # index_col = 0 意思是将第一列设置为索引值
    if ds.empty:
        return ds

    day_pm = ds[ds['type'] == 'PM2.5_24h'].iloc[:, 2:]
    return day_pm.iloc[-1]   # 返回最后一个小时的24小时滑动平均




# 开始处理每年的数据
def dealYearO3(dir='F:/2_Joint_Estimation/Source Data/Site_label', year='站点_20190101-20191231'):  # 即'E:/3_Atmos/Source Data/Site_label'， ”站点_20150101-20151231“
    yeardir = os.path.join(dir, year)
    file = os.listdir(yeardir)
    # 选择数据如：china_sites_20190101.csv
    file = [f for f in os.listdir(yeardir) if f.startswith('china')]  # 这里进行排序，并过滤掉不符合规则的文件或文件夹

    # 读取每年第一天的值作为样本参考
    sample = dealDayO3(os.path.join(yeardir, file[0]))  # 判断一年中每天的站点是否相同

    """
    (1)python根据年月日到的相应的天数：
    testtime = time.strptime('20141221', "%Y%m%d")， testtime.tm_yday
    strptime得到的是一个日期时间对象time，它有日期和时间组件。格式化输出：time.strftime('%Y%m%d', testtime)
    (2)python根据年天数得到相应的月日：
    testtime = time.strptime('2014133', "%Y%j'),
    (3)由于某些年份的数据存在某日的缺失，所以需要进行额外的处理，即判断日期是否连续
    双指针，指针p1指向当前的日期，p2指向对应的天数；从(20140513, 133) 迭代到 (20141231, 365)
    在迭代的过程中，如果p2得到的时间小于当前时间，那么插入一个p2时间的空series, 直到对应的两个时间字符串相等。
    """

    tstruct2 = time.strptime(year.split('_')[1][:8], '%Y%m%d')  # 读取第一个时间结构体，以及对应的天数
    tday = tstruct2.tm_yday  # 根据目录如'站点_20150101-20151231'确定的该年的第一天
    for i in tqdm(range(len(file))):
        day_out = dealDayO3(os.path.join(yeardir, file[i]))

        if i == 0:
            # 处理文件时间
            tstruct = time.strptime(file[i].split('_')[2][:8], '%Y%m%d')  # 如 'china_sites_20190101.csv'
            if tstruct != tstruct2:  # 处理有缺失的情况，即目录的'站点_20150101'与其中的'china_sites_20150102'不一致
                while tstruct != tstruct2:  # 直到文件夹中的第一个时间文件
                    day_out = pd.DataFrame(index=sample.index, columns=[time.strftime('%Y%m%d', tstruct2)])
                    year_out = day_out
                    tday += 1
                    tstruct2 = time.strptime(file[i].split('_')[2][:4] + str(tday), '%Y%j')  # 年份+在年中的天数
                day_out = dealDayO3(os.path.join(yeardir, file[i]))
                year_out = pd.concat([year_out, day_out], axis=1)
            else:
                tday = tstruct.tm_yday
                year_out = day_out
        else:
            tday += 1
            tstruct2 = time.strptime(file[i].split('_')[2][:4] + str(tday), '%Y%j')  # 这里是根据年份、连续天数得到的时间结构体
            tstruct = time.strptime(file[i].split('_')[2][:8], '%Y%m%d')  # 这里是根据文件名得到的时间结构体
            while tstruct != tstruct2:  # 如果时间不连续，添加空的时间列
                day_out = pd.DataFrame(index=sample.index, columns=[time.strftime('%Y%m%d', tstruct2)])
                year_out = pd.concat([year_out, day_out], axis=1)
                tday += 1
                tstruct2 = time.strptime(file[i].split('_')[2][:4] + str(tday), '%Y%j')  # 这里是连续得到的时间结构体
            if day_out.empty:
                day_out = pd.DataFrame(index=sample.index, columns=[file[i].split('_')[2][:8]])
            # 注意，一年内每天的站点个数并不相同
            # 但是DataFrame存在广播机制，可以将某天不存在站点的数据补空
            year_out = pd.concat([year_out, day_out], axis=1)
    return year_out

yearpath = '站点_20190101-20191231'
year_o3 = dealYearO3(year = yearpath)


def dealYearPM(dir='F:/2_Joint_Estimation/Source Data/Site_label', year='站点_20150101-20151231'):  # 即'E:/3_Atmos/Source Data/Site_label'， ”站点_20150101-20151231“
    yeardir = os.path.join(dir, year)
    file = os.listdir(yeardir)
    # 选择数据如：china_sites_20190101.csv
    file = [f for f in os.listdir(yeardir) if f.startswith('china')]  # 这里进行排序，并过滤掉不符合规则的文件或文件夹

    # 读取每年第一天的值作为样本参考
    sample = dealDayPM(os.path.join(yeardir, file[0]))  # 判断一年中每天的站点是否相同

    """
    (1)python根据年月日到的相应的天数：
    testtime = time.strptime('20141221', "%Y%m%d")， testtime.tm_yday
    strptime得到的是一个日期时间对象time，它有日期和时间组件。格式化输出：time.strftime('%Y%m%d', testtime)
    (2)python根据年天数得到相应的月日：
    testtime = time.strptime('2014133', "%Y%j'),
    (3)由于某些年份的数据存在某日的缺失，所以需要进行额外的处理，即判断日期是否连续
    双指针，指针p1指向当前的日期，p2指向对应的天数；从(20140513, 133) 迭代到 (20141231, 365)
    在迭代的过程中，如果p2得到的时间小于当前时间，那么插入一个p2时间的空series, 直到对应的两个时间字符串相等。
    """

    tstruct2 = time.strptime(year.split('_')[1][:8], '%Y%m%d')  # 读取第一个时间结构体，以及对应的天数
    tday = tstruct2.tm_yday  # 根据目录如'站点_20150101-20151231'确定的该年的第一天
    for i in tqdm(range(len(file))):
        day_out = dealDayPM(os.path.join(yeardir, file[i]))

        if i == 0:
            # 处理文件时间
            tstruct = time.strptime(file[i].split('_')[2][:8], '%Y%m%d')  # 如 'china_sites_20190101.csv'
            if tstruct != tstruct2:  # 处理有缺失的情况，即目录的'站点_20150101'与其中的'china_sites_20150102'不一致
                while tstruct != tstruct2:  # 直到文件夹中的第一个时间文件
                    day_out = pd.DataFrame(index=sample.index, columns=[time.strftime('%Y%m%d', tstruct2)])
                    year_out = day_out
                    tday += 1
                    tstruct2 = time.strptime(file[i].split('_')[2][:4] + str(tday), '%Y%j')  # 年份+在年中的天数
                day_out = dealDayPM(os.path.join(yeardir, file[i]))
                year_out = pd.concat([year_out, day_out], axis=1)
            else:
                tday = tstruct.tm_yday
                year_out = day_out
        else:
            tday += 1
            tstruct2 = time.strptime(file[i].split('_')[2][:4] + str(tday), '%Y%j')  # 这里是根据年份、连续天数得到的时间结构体
            tstruct = time.strptime(file[i].split('_')[2][:8], '%Y%m%d')  # 这里是根据文件名得到的时间结构体
            while tstruct != tstruct2:  # 如果时间不连续，添加空的时间列
                day_out = pd.DataFrame(index=sample.index, columns=[time.strftime('%Y%m%d', tstruct2)])
                year_out = pd.concat([year_out, day_out], axis=1)
                tday += 1
                tstruct2 = time.strptime(file[i].split('_')[2][:4] + str(tday), '%Y%j')  # 这里是连续得到的时间结构体
            if day_out.empty:
                day_out = pd.DataFrame(index=sample.index, columns=[file[i].split('_')[2][:8]])
            # 注意，一年内每天的站点个数并不相同
            # 但是DataFrame存在广播机制，可以将某天不存在站点的数据补空
            year_out = pd.concat([year_out, day_out], axis=1)
    return year_out

# yearpath = '站点_20190101-20191231'
# year_pm = dealYearPM(year = yearpath)
# year_pm.to_csv(r'E:\3_Atmos\dataset\site_pm_2019.csv')

# 该函数将每年的环境站点的文件，生成相关污染物每年的日数据, 包含执行单个dealYearLabel函数
# os.path.isdir(path)  来判别该路径是否为文件夹，使用os.path.isfile(path)来判定是否为文件
# 避免使用相对路径，以提升代码的可移植性
def dealMultiYearLabel(direct='E:/3_Atmos/Source Data/Site_label'):
    yearDir = os.listdir(direct)  # 如 '站点_20140513-20141231'
    yearDir = [f for f in os.listdir(direct) if os.path.isdir(os.path.join(direct, f)) and f.startswith('站点')]
    yearDir.sort(key=lambda x: int(x.split('_')[1][:5]))  # 格式 ”站点_20150101-20151231“
    print(yearDir, len(yearDir))
    for i in range(len(yearDir)):  # len(yearDir), 一共8年的数据    , len(yearDir)
        print(yearDir[i])
        year_out = dealYearLabel(direct, yearDir[i])
        year_out.to_csv('../Site_label/' + yearDir[i] + '.csv')


# 第一步，生成每年对应的O3数据；
# dealMultiYearLabel()

# 第二部（可选），将每年的O3数据整合起来，得到某些站点的长时序数据
def integrate(direct, file):
    subindex = [[]] * len(file)
    for i in range(len(file)):
        # 判断时间是否连续
        tstruct = time.strptime(file[i].split('_')[1][:8], '%Y%m%d')
        tstruct2 = time.strptime(file[i].split('_')[1][9:17], '%Y%m%d')
        ds = pd.read_csv(os.path.join(direct, file[i]), index_col=0)  # index_col = 0 意思是将第一列设置为索引值
        print(file[i], ds.shape, tstruct2.tm_yday - tstruct.tm_yday + 1)
        subindex[i] = ds.index

        if i == 0:
            long_label = ds
            startday = file[i].split('_')[1].split('-')[0]
        else:
            if i == len(file) - 1:
                endday = file[i].split('_')[1].split('-')[1]

            # 这里合并的时候，每年的索引可能出现不一致的情况
            long_label = pd.concat([long_label, ds], axis=1)
    long_label.to_csv('../Site_label/全国_' + startday + '-' + endday)
    print(long_label.shape)


def debug_nan():
    # 这里某些年份王**提供的数据中，可能缺少某天的数据
    direct = '../Site_label'
    file = os.listdir(direct)
    file = [f for f in os.listdir(direct) if f.endswith('.csv') and f.startswith('站点')]  # 通过字符串的startswith和endswith函数匹配相关的文件
    file.sort(key = lambda x: int(x.split('_')[1][:4]))
    # integrate(direct, file)  # 经过这里的步骤之后，我们得到2014-2021各年的数据


    # 判定合成的csv是否包含每个子csv文件
    # np.nan， 使用pd.isnull()函数进行处理， pd.isna()的效果一致；与notnull()的结果相反。
    for i in range(len(file)):
        ds1 = pd.read_csv(os.path.join(direct, file[i]), index_col=0).fillna(-9999)  # index_col = 0 意思是将第一列设置为索引值
        print(file[i], ds1.shape)
        # ds2 = long_label.loc[ds1.index, ds1.columns].fillna(-9999)
        # print(file[i], (ds1 == ds2).all().all())

