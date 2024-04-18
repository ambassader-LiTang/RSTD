import math
import os
import cv2
from torchvision import datasets, transforms
import json
from PIL import Image
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.wkt import loads
import matplotlib.image as mpimg
from sklearn.cluster import SpectralClustering

from sklearn.cluster import OPTICS
from shapely.geometry import MultiPolygon, Polygon


from geopy.distance import great_circle
from scipy import stats
import pysal as ps
from scipy.spatial import distance
from collections import defaultdict
from pysal.explore import esda
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from pysal.lib import weights
import matplotlib.patches as patches
# 定义数据根目录
data_root_dir = r"E:\Dataset\xview2\xview2\images"
from sklearn.cluster import DBSCAN
from shapely.ops import unary_union

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
Image.MAX_IMAGE_PIXELS = 1000000000
#

centercropL=transforms.CenterCrop((1024,1024))
centercropS=transforms.CenterCrop((512,512))
def classify_bbox(bw, bh, w, h):
    # 计算包围盒的最小边与图像长或宽的比例
    ratio = min(bw / w, bh / h)

    # 根据比例确定类别
    if ratio >= 1 / 2:
        return 0
    elif ratio >= 1 / 4:
        return 1
    elif ratio >= 1 / 8:
        return 2
    elif ratio >= 1 / 16:
        return 3
    else:
        return 4
def get_position(bx, by, w, h):
    # 左上角
    if bx < 0.375 * w and by < 0.375 * h:
        return "Top Left"
    # 左下角
    elif bx < 0.375 * w and by > 0.625 * h:
        return "Bottom Left"
    # 右上角
    elif bx > 0.625 * w and by < 0.375 * h:
        return "Top Right"
    # 右下角
    elif bx > 0.625 * w and by > 0.625 * h:
        return "Bottom Right"
    # 上方
    elif bx >= 0.375 * w and bx <= 0.625 * w and by < 0.375 * h:
        return "Top"
    # 下方
    elif bx >= 0.375 * w and bx <= 0.625 * w and by > 0.625 * h:
        return "Bottom"
    # 左方
    elif bx < 0.375 * w and by >= 0.375 * h and by <= 0.625 * h:
        return "Left"
    # 右方
    elif bx > 0.625 * w and by >= 0.375 * h and by <= 0.625 * h:
        return "Right"
    # 中心
    else:
        return "Center"

# 计算局部莫兰指数
def merge_nearby_points(points, merge_radius):
    merged_points = []
    while len(points) > 0:
        # 从点集中选择一个点作为当前基准点
        base_point = points.pop(0)
        # 找到距离基准点小于合并半径的所有点
        nearby_points = [p for p in points if euclidean_distances([base_point], [p])[0][0] <= merge_radius]
        # 计算合并后的点，这里简单地取平均值
        merged_point = np.mean([base_point] + nearby_points, axis=0)
        # 将合并后的点添加到新的点集中
        merged_points.append(merged_point)
        # 从原始点集中移除已经合并的点
        for p in nearby_points:
            points.remove(p)
    return merged_points

def update_points(points, min_samples=5, xi=0.05, merge_radius=10):
    # 合并距离小于指定范围的点
    merged_points = merge_nearby_points(points, merge_radius)
    merged_points_array = np.array(merged_points)

    # 使用OPTICS聚类算法
    optics = OPTICS(min_samples=min_samples, xi=xi)
    optics.fit(merged_points_array)

    updated_points = []

    # 遍历每个簇
    for label in np.unique(optics.labels_):
        # 如果标签为 -1，表示噪声点，不处理
        if label == -1:
            continue
        # 找到当前簇的所有点
        cluster_points = merged_points_array[optics.labels_ == label]
        # 计算当前簇的中心点
        cluster_center = np.mean(cluster_points, axis=0)
        # 将当前簇的所有点替换为中心点
        updated_points.append(cluster_center.tolist())

    return updated_points
def update_points2(points, eps=10, min_samples=1):
    # 将点集转换为numpy数组
    # 将点集转换为numpy数组
    points_array = np.array(points)

    # 使用DBSCAN聚类算法
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(points_array)

    # 获取每个簇的标签和唯一标签列表
    labels = dbscan.labels_
    unique_labels = np.unique(labels)

    # 初始化字典来存储每个簇的中心点和点的索引列表
    clusters = {}

    # 遍历每个簇
    for label in unique_labels:
        # 如果标签为 -1，表示噪声点，不处理
        if label == -1:
            continue
        # 找到当前簇的所有点
        cluster_points = points_array[labels == label]
        # 计算当前簇的中心点
        cluster_center = np.mean(cluster_points, axis=0)
        # 获取当前簇的点在原始数据中的索引列表
        cluster_indices = np.where(labels == label)[0]
        # 将当前簇的中心点和点的索引列表存储到字典中
        clusters[label] = {'center': cluster_center.tolist(), 'indices': cluster_indices.tolist()}

    return clusters


def crop_center(img,cx,cy,size):



    (w,h)=img.size
    left=int(w-size if cx+size/2>w else (0 if cx-size/2<0 else cx-size/2))

    top=int(h-size if cy+size/2>h else (0 if cy-size/2<0 else cy-size/2))

    right= int(left+size)
    bottom=int(top+size)

    if right>w or bottom > h:#超了
        return None,None,None

    newimg=img.crop((left,top,right,bottom))




    return newimg,left,top

# 创建数据加载器
image_num=0
iL=0
iS=0
print("开始处理")
savepath=r'E:\Dataset\xview2\xview2'
for category_dir in os.listdir(data_root_dir):
    print(f'类别{category_dir}')
    # 获取类别名称


    # 遍历类别文件夹中的所有图片文件夹序列
    # for image_dir in os.listdir(os.path.join(data_root_dir, category_dir,'images')):
    #     # 获取图片文件夹序列名称
    #     image_dir_name = os.path.join(data_root_dir, category_dir,'images', image_dir)
    if True:
        image_dir=category_dir
        image_dir_name=os.path.join(data_root_dir, category_dir)
        # dataset = gdal.Open(image_dir_name)
        # #读取图像数据
        # band1 = dataset.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        # band2 = dataset.GetRasterBand(2).ReadAsArray().astype(np.uint8)
        # band3 = dataset.GetRasterBand(3).ReadAsArray().astype(np.uint8)
        #
        # zero_pixels_ratio = np.count_nonzero(band1 == 0) / band1.size
        #
        # #使用阈值来判断是否有黑色覆盖
        # if zero_pixels_ratio > 0.05:
        #     print(f"图像文件 {image_dir} 有部分区域是全黑色的")
        #     continue
        #
        #
        #
        #
        # #将三个波段堆叠成RGB图像
        # rgb_image = Image.merge("RGB", (Image.fromarray(band1), Image.fromarray(band2), Image.fromarray(band3)))

        #json_path = os.path.join(data_root_dir, category_dir, 'labels', image_dir.split('.')[0] + '.json')

        json_path=os.path.join(data_root_dir.split(r'images')[0],'labels',image_dir.split('.')[0] + '.json')

        with open(json_path, "r") as f:
            metadata = json.load(f)

        #save_path = os.path.join(savepath, 'images')
        #os.makedirs(save_path, exist_ok=True)

        #rgb_image.save(os.path.join(save_path, image_dir.split('.')[0] + '.jpg'))


        json_file_path = os.path.join(savepath, 'labels')
        os.makedirs(json_file_path, exist_ok=True)

        json_file_path = os.path.join(json_file_path, image_dir.split('.')[0] + '.json')

        btypes={}

        centers=[]
        attrs=[]
        bounds=[]

        binfo ={}

        for i,bbox in enumerate(metadata['features']['xy']):
            wkt=bbox['wkt']
            polygon = loads(wkt)
            center=tuple(polygon.centroid.coords[0])

            bound = polygon


            attr=bbox['properties']['feature_type']



            typelist= btypes.get(attr,[])
            typelist.append(i)
            btypes.update({attr:typelist})


            # 计算多边形的面积
            attrs.append(attr)
            centers.append(center)
            bounds.append(bound)

        # image = rgb_image
        # plt.figure(figsize=(8, 6))
        # plt.imshow(image)  # 显示图片


        for bkey,bvalue in btypes.items():

            a_centers =  np.array(centers)[bvalue].tolist()

            center2 = update_points2(a_centers, 128, 1)
            bvalue=np.array(bvalue)
            bboxs =[]

            cur_binfo_class={}
            cur_binfo={bkey:cur_binfo_class}
            for ind in center2.values():


                indices=bvalue[ind['indices']]
                # bounds_union = unary_union([bounds[i] for i in indices])
                # b=bounds_union.bounds


                A=MultiPolygon([bounds[i] for i in indices])
                polygons = A.buffer(0)

                # 获取多边形集合的边界并集
                bound_union = polygons.union(polygons)
                b=bound_union.bounds
                bboxs.append(b)
                # rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], edgecolor='r', facecolor='none')

                pos=get_position(ind['center'][0],ind['center'][1],1024,1024)

                if len(indices)>10:
                    poslist= cur_binfo_class.get('large',{'poslist':[],'size':[]})

                    poslist['poslist'].append(pos)
                    poslist['size'].append(classify_bbox( b[2] - b[0], b[3] - b[1],1024,1024))
                    cur_binfo_class.update({'large':poslist})

                elif len(indices)>1:
                    poslist= cur_binfo_class.get('medium',{'poslist':[],'size':[]})

                    poslist['poslist'].append(pos)
                    poslist['size'].append(classify_bbox( b[2] - b[0], b[3] - b[1],1024,1024))
                    cur_binfo_class.update({'medium':poslist})
                else:
                    poslist= cur_binfo_class.get('single',{'poslist':[],'size':[]})

                    poslist['poslist'].append(pos)
                    poslist['size'].append(classify_bbox( b[2] - b[0], b[3] - b[1],1024,1024))
                    cur_binfo_class.update({'single':poslist})

            binfo.update(cur_binfo)





        #center2=update_points(centers,2, merge_radius=10)
        #center2 = update_points2(centers, 128, 1)

        #
        # for center in centers:
        #     plt.scatter(center[0], center[1], color='blue')
        # for c in center2.values():
        #     print('s')
        #     plt.scatter(c['center'][0],c['center'][1], color='red',)
        # plt.show()


        # # 创建一个空白画布
        # plt.figure(figsize=(8, 6))
        # plt.gca().set_facecolor('black')  # 设置背景色为黑色
        #
        # # 遍历中心点并绘制白色点
        # for center in centers:
        #     plt.scatter(center[0], center[1], color='white')
        #
        # # 设置坐标轴标签颜色为白色
        # plt.tick_params(axis='x', colors='white')
        # plt.tick_params(axis='y', colors='white')
        #
        # plt.show()

        with open(json_file_path, "w") as json_file:
            metadata['binfo']=binfo
            json.dump(metadata, json_file,indent=4)
            image_num+=1
            print(image_num)


