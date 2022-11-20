from osgeo import gdal
import os
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt

def tif2nc(path, time='20190101'):
    """
    输入时间的形式：'%Y-%m-%d'，匹配GEE的保存格式
    返回一个nc文件，包含数据，经纬度和时间
    GeoTransform 的含义：
        影像左上角横坐标：im_geotrans[0]，对应经度
        影像左上角纵坐标：im_geotrans[3]，对应纬度

        遥感图像的水平空间分辨率(纬度间隔)：im_geotrans[5]
        遥感图像的垂直空间分辨率(经度间隔)：im_geotrans[1]
        通常水平和垂直分辨率相等

        如果遥感影像方向没有发生旋转，即上北下南，则 im_geotrans[2] 与 im_geotrans[4] 为 0

    计算图像地理坐标：
        若图像中某一点的行数和列数分别为 row 和 column，则该点的地理坐标为：
            经度：xGeo = im_geotrans[0] + col * im_geotrans[1] + row * im_geotrans[2]
            纬度：yGeo = im_geotrans[3] + col * im_geotrans[4] + row * im_geotrans[5]
    """

    """
    GetRasterBand(bandNum)，选择要读取的波段数，bandNum 从 1 开始
    ReadAsArray(xoff, yoff, xsize, ysize)，一般就按照下面这么写，偏移量都是 0 ，返回 ndarray 数组
    """
    data = gdal.Open(path)
    im_width = data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
    im_height = data.RasterYSize  # 获取高度，数组第一维，上下方向元素长度，代表纬度范围
    im_bands = data.RasterCount  # 波段数
    im_geotrans = data.GetGeoTransform()  # 获取仿射矩阵，含有 6 个元素的元组
    im_proj = data.GetProjection()  # 获取地理信息
    # 按照波段获取数据
    # data.GetRasterBand(1).GetScale(), data.GetRasterBand(1).GetOffset()
    scale = 0.001
    im_data = data.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=im_width, win_ysize=im_height)*scale  # 默认第一个波段
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]
    # 获取所有波段的数据
    # im_nc = xr.DataArray(im_data, coords=[im_lat, im_lon], dims=['lat', 'lon'])

    im_nc = xr.DataArray(im_data[np.newaxis, :, :].astype('float32'), coords=[[datetime.strptime(time, '%Y-%m-%d')], im_lat, im_lon], dims=['time', 'lat', 'lon'])
    return im_nc


def tif2info(path):
    """

    :param path: 文件的路径
    :return: 文件的经纬度信息
    """

    data = gdal.Open(path)
    im_width = data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围, (72.5, 136.25) -> 7098
    im_height = data.RasterYSize  # 获取高度，数组第一维，上下方向元素长度，代表纬度范围, (54.5, 18) -> 4064
    im_bands = data.RasterCount  # 波段数
    im_geotrans = data.GetGeoTransform()  # 获取仿射矩阵，含有 6 个元素的元组
    # 产生文件的经纬度
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]
    return im_bands, im_lat, im_lon


Dir = 'F:\\2_联合估算\\Source Data\\GEE_AOD2019'
file = os.listdir(Dir)
file = [f for f in file if f.endswith('01.tif')]
file.sort(key=lambda x: int(x.split('.')[0][-5:-3]))
fpath = [os.path.join(Dir, i) for i in file]

ex_nc = [tif2nc(path=f, time=f.split('.')[0][-10:]) for f in fpath]
[f.to_netcdf('gee-aod_'+f.time.data.astype(str)[0][:10]+'.nc') for f in ex_nc]

def plot():
    """
    遥感反演的气溶胶光学厚度本质上是整个大气的消光系数，可以近似地理解为与大气顶层的太阳辐射能相比，地面得到的辐射能占其几分之一；
    由于我们想要的是大气污染程度而不是到底接收到多少能量，所以取了一个接近无污染的时间，把它的AOD设为0，记下这时的大气条件；
    再找这个地方史上污染最严重时刻，把它的AOD设为1，同样记录大气条件。然后把这套标量应用到相关的辐射传输模型里，大家都用，
    这样子就跟其它指数一样可以在0到1之间变动了。
    :return:
    """
    for i in range(len(ex_nc)):
        ex_nc[i].plot(cmap = "Spectral_r", vmin=0, vmax=1)  # Spectral的互逆配色
        plt.savefig('gee-aod_'+ex_nc[i].time.data.astype(str)[0][:10]+'.png')
        plt.show()

"观察GEE中导出的两种方式，对应的经纬度是否相同。经过分析，两种坐标系导出的形式相同。（绘图3857，保存4326）"
def bool_RDS():
    """
    coordinate reference system of 4326 and 3857
    :return:
    """

    file_epsg = 'F:\\2_联合估算\\Source Data\\GEE_AOD2019\\AOD_2019-03-01_EPSG3857.tif'
    data_4326, data_3857 = gdal.Open(fpath[0]), gdal.Open(file_epsg)
    print('经纬度', data_4326.RasterXSize, data_4326.RasterYSize, data_3857.RasterXSize, data_3857.RasterYSize)
    print('仿射矩阵', data_4326.GetGeoTransform(), data_3857.GetGeoTransform())
    print('数据形状', data_4326.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=data_4326.RasterXSize,
                                                 win_ysize=data_4326.RasterYSize).shape,
          data_3857.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=data_3857.RasterXSize,
                                                 win_ysize=data_3857.RasterYSize).shape)
    print('地理信息', data_4326.GetProjection(), data_3857.GetProjection() )
    pro_4326 = gdal.Open(fpath[0]).GetGeoTransform()
    pro_3857 = gdal.Open(file_epsg).GetGeoTransform()


