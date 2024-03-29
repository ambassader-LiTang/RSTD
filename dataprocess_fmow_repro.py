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
data_root_dir = r"F:\数据集\fmow_flit"
meta_root_dir = r"F:\数据集\fmow_meta"





# 创建数据加载器
i=0
iL=0
iS=0
print("开始处理")
for category_dir in os.listdir(data_root_dir):

    print(f'类别{category_dir}')
    # 获取类别名称
    category_name = category_dir

    # 遍历类别文件夹中的所有图片文件夹序列
    for image_dir in os.listdir(os.path.join(data_root_dir, category_dir)):
        # 获取图片文件夹序列名称
        image_dir_name = image_dir

        #获取图片名字
        for img in  os.listdir(os.path.join(data_root_dir, category_dir,image_dir)):

            #排除msrgb和json
            if img.endswith('json') or img.split('.')[0].endswith('msrgb'):
                continue

            #获得图片路径
            image_path = os.path.join(data_root_dir, category_dir, image_dir,img)


            #获得元数据路径
            json_path=os.path.join(meta_root_dir, category_dir, image_dir,img.split('.')[0]+'.json')



            #打开元数据


            image = Image.open(image_path)
            imgarry=np.array(image)

            zero_pixels_ratio = np.count_nonzero(imgarry == 0) / (imgarry.size)

            # 使用阈值来判断是否有黑色覆盖
            if zero_pixels_ratio > 0.05:
                print(f"图像文件 {image_dir} 有部分区域是全黑色的")
                os.remove(image_path)
                os.remove(json_path)
                continue
            else:
                if category_dir=='1024':
                    iL+=1
                else:
                    iS+=1
                i+=1
            if i%100==0:
                print(i)


print(i)
print(iL)
print(iS)





        # 获取图片路径和元数据路径
        # image_path = os.path.join(data_root_dir, category_dir, image_dir, f"{image_dir_name}_rgb.jpg")
        # json_path = os.path.join(data_root_dir, category_dir, image_dir, f"{image_dir_name}_rgb.json")
        #
        # # 读取图片
        # image = cv2.imread(image_path)
        #
        # # 读取元数据
        # with open(json_path, "r") as f:
        #     metadata = json.load(f)
        #
        # # 对图片和元数据进行处理
        # # ...
        #
        # # 打印图片和元数据信息
        # print(f"类别: {category_name}")
        # print(f"图片文件夹序列: {image_dir_name}")
        # print(f"图片路径: {image_path}")
        # print(f"元数据: {metadata}")
        #


# 调用函数示例
