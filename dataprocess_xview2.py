import math
import os
import cv2
from torchvision import datasets, transforms
import json
from PIL import Image
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

# 定义数据根目录
data_root_dir = r"E:\Dataset\xview2\geotiffs"

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
Image.MAX_IMAGE_PIXELS = 1000000000
#

centercropL=transforms.CenterCrop((1024,1024))
centercropS=transforms.CenterCrop((512,512))




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
i=0
iL=0
iS=0
print("开始处理")
savepath=r'E:\Dataset\xview2\xview2'
for category_dir in os.listdir(data_root_dir):
    print(f'类别{category_dir}')
    # 获取类别名称


    # 遍历类别文件夹中的所有图片文件夹序列
    for image_dir in os.listdir(os.path.join(data_root_dir, category_dir,'images')):
        # 获取图片文件夹序列名称
        image_dir_name = os.path.join(data_root_dir, category_dir,'images', image_dir)
        dataset = gdal.Open(image_dir_name)
        # 读取图像数据
        band1 = dataset.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        band2 = dataset.GetRasterBand(2).ReadAsArray().astype(np.uint8)
        band3 = dataset.GetRasterBand(3).ReadAsArray().astype(np.uint8)

        zero_pixels_ratio = np.count_nonzero(band1 == 0) / band1.size

        # 使用阈值来判断是否有黑色覆盖
        if zero_pixels_ratio > 0.05:
            print(f"图像文件 {image_dir} 有部分区域是全黑色的")
            continue




        # 将三个波段堆叠成RGB图像
        rgb_image = Image.merge("RGB", (Image.fromarray(band1), Image.fromarray(band2), Image.fromarray(band3)))

        json_path = os.path.join(data_root_dir, category_dir, 'labels', image_dir.split('.')[0] + '.json')

        with open(json_path, "r") as f:
            metadata = json.load(f)

        save_path = os.path.join(savepath, 'images')
        os.makedirs(save_path, exist_ok=True)

        rgb_image.save(os.path.join(save_path, image_dir.split('.')[0] + '.jpg'))


        json_file_path = os.path.join(savepath, 'labels')
        os.makedirs(json_file_path, exist_ok=True)

        json_file_path = os.path.join(json_file_path, image_dir.split('.')[0] + '.json')

        with open(json_file_path, "w") as json_file:
            json.dump(metadata, json_file)


