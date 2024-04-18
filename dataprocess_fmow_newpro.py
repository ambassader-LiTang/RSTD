import math
import os
import random

import cv2
from torchvision import datasets, transforms
import json
from PIL import Image
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

# 定义数据根目录
data_root_dir = r"F:\数据集\fmow\train"

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
Image.MAX_IMAGE_PIXELS = 1000000000
#

centercropL=transforms.CenterCrop((1024,1024))
centercropS=transforms.CenterCrop((512,512))



#中1，上2，下3，左4，右5，左上6，右上7，左下8，右下9
def crop_center(img,cx,cy,size):

    (w,h)=img.size
    left=int(w-size if cx+size/2>w else (0 if cx-size/2<0 else cx-size/2))

    top=int(h-size if cy+size/2>h else (0 if cy-size/2<0 else cy-size/2))

    right= int(left+size)
    bottom=int(top+size)

    if right>w or bottom > h or left<0 or top<0:#超了
        return None,None,None
    newimg=img.crop((left,top,right,bottom))
    return newimg,left,top

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


def resize_and_crop(img, size, bx, by, bw, bh):
    # 计算压缩比例
    (w, h) = img.size
    ratio = max(size / w, size / h)

    # 计算压缩后的图像尺寸
    new_w = int(np.ceil(w * ratio))
    new_h = int(np.ceil(h * ratio))

    # 计算包围盒的更新坐标和大小
    new_bx = int(bx * ratio)
    new_by = int(by * ratio)
    new_bw = int(bw * ratio)
    new_bh = int(bh * ratio)

    # 计算裁剪后的图像范围

    crop_left=int(new_w/2-size/2)
    crop_top=int(new_h/2-size/2)
    crop_right=crop_left+size
    crop_bottom=crop_top+size
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # 如果裁剪区域超出图像范围，则返回None
    if new_bx< crop_left:

        crop_left=new_bx
        crop_right=crop_left+size
        if crop_right>new_w:
            return None,None,None,None,None,None
    elif new_bx+new_bw> crop_right:
        crop_right=new_bx+new_bw
        crop_left=crop_right-size
        if crop_left<0:
            return None,None,None,None,None,None

    if new_by < crop_top:
        crop_top=new_by
        crop_bottom=crop_top+size
        if crop_bottom>new_h:
            return None,None,None,None,None,None


    elif new_by+new_bh> crop_bottom:
        crop_bottom= new_by+new_bh
        crop_top=crop_bottom-size
        if crop_top<0:
            return None,None,None,None,None,None

    # 裁剪图像
    new_bx=new_bx-crop_left
    new_by=new_by-crop_top
    cropped_image = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # 返回裁剪后的图像、压缩比例和更新后的包围盒坐标和大小
    return cropped_image, ratio, new_bx, new_by, new_bw, new_bh


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
            json_path=os.path.join(data_root_dir, category_dir, image_dir,img.split('.')[0]+'.json')



            #打开元数据
            with open(json_path, "r") as f:
                 metadata = json.load(f)


            w= int(metadata['img_width']) #得到尺寸
            h= int(metadata['img_height'])


            bounding_box_list=metadata['bounding_boxes'][0]
            gsd=metadata['gsd']
            utm=metadata['utm']
            country=metadata['country_code']
            time=metadata['timestamp']
            scan_direction=metadata['scan_direction']

            bx=bounding_box_list['box'][0]
            by=bounding_box_list['box'][1]
            bw=bounding_box_list['box'][2]
            bh=bounding_box_list['box'][3]

            #img=img[:-4]+'_val'+img[-4:]
            if min(w,h)>=1024:
                if max(bw,bh)>=1024: #其中一个大于1024
                    if max(bw,bh)>5000: #其中一个大于4096
                        continue
                    else:
                        try:  # 如果打不开，则下一个
                            image = Image.open(image_path)
                        except FileExistsError:
                            continue



                        if random.random()< 0.2:
                            side = max(bw, bh)
                            cx = int(bx + bw / 2)
                            cy = int(by + bh / 2)
                            image, left, top = crop_center(image, cx, cy, side)
                            if not image:  # 裁剪失败，不够大，就走
                                continue
                            bx = bx - left
                            by = by - top

                            scale = 1024 / side
                            image= image.resize((1024, 1024), Image.LANCZOS)

                            bx = math.floor(scale * bx)
                            by = math.floor(scale * by)
                            bw = math.floor(scale * bw)
                            bh = math.floor(scale * bh)

                            pos = get_position(cx, cy, 1024, 1024)
                            size = classify_bbox(bw, bh, 1024, 1024)
                        else:
                            image,scale,bx,by,bw,bh=resize_and_crop(image,1024,bx,by,bw,bh)
                            if not image or min(bw,bh)<10:
                                continue
                            cx = int(bx + bw / 2)
                            cy = int(by + bh / 2)
                            pos = get_position(cx, cy, 1024, 1024)
                            size = classify_bbox(bw, bh, 1024, 1024)


                        save_path=os.path.join( r"F:\数据集\fmow_fli",'1024',category_name)
                        os.makedirs(save_path, exist_ok=True)
                        image.save(os.path.join(save_path,img))
                        dict={}
                        bounding_box_list['box'][0]=bx
                        bounding_box_list['box'][1]=by
                        bounding_box_list['box'][2]=bw
                        bounding_box_list['box'][3]=bh

                        dict.update({'size':size})
                        dict.update({'pos':pos})

                        dict.update({'gsd':gsd/scale})
                        dict.update({'utm':utm})
                        dict.update({'country_code': country})
                        dict.update({'time':time})
                        dict.update({'scan_direction':scan_direction})
                        dict.update({'bounding_box':bounding_box_list})
                        json_file_path =  os.path.join(r"F:\数据集\fmow_met",'1024',category_name)
                        os.makedirs(json_file_path, exist_ok=True)
                        json_file_path=os.path.join(json_file_path,img.split('.')[0]+'.json')
                        # 将字典写入 JSON 文件
                        with open(json_file_path, "w") as json_file:
                            json.dump(dict, json_file)
                        i+=1
                        iL+=1
                else:
                    try:  # 如果打不开，则下一个
                        image = Image.open(image_path)
                    except FileExistsError:
                        continue
                    if random.random()<=0.2:
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                        newimg, left, top = crop_center(image, cx, cy, 1024)

                        if newimg == None:
                            continue
                        bx=bx-left
                        by=by-top
                        if bx<0:
                            bx=0
                            bw=1024
                        if by<0:
                            by=0
                            bh=1024
                    else:
                        newimg, scale, bx, by, bw, bh = resize_and_crop(image, 1024, bx, by, bw, bh)
                        if not image or min(bw, bh) < 10:
                            continue
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                        pos = get_position(cx, cy, 1024, 1024)
                        size = classify_bbox(bw, bh, 1024, 1024)




                    save_path = os.path.join(r"F:\数据集\fmow_fli", '1024', category_name)
                    os.makedirs(save_path, exist_ok=True)
                    newimg.save(os.path.join(save_path, img))
                    dict = {}
                    bounding_box_list['box'][0] = bx
                    bounding_box_list['box'][1] = by
                    bounding_box_list['box'][2] = bw
                    bounding_box_list['box'][3] = bh
                    dict.update({'gsd': gsd })
                    dict.update({'utm': utm})
                    dict.update({'country_code': country})
                    dict.update({'time': time})
                    dict.update({'scan_direction': scan_direction})
                    dict.update({'bounding_box': bounding_box_list})
                    json_file_path = os.path.join(r"F:\数据集\fmow_met", '1024', category_name)
                    os.makedirs(json_file_path, exist_ok=True)
                    json_file_path = os.path.join(json_file_path, img.split('.')[0] + '.json')
                    # 将字典写入 JSON 文件
                    with open(json_file_path, "w") as json_file:
                        json.dump(dict, json_file)
                    i += 1
                    iL +=1

            elif min(w,h)>=512:
                if max(bw,bh)>=512: #其中一个大于512
                    if max(bw,bh)>1024: #其中一个大于1024
                        continue
                    else:
                        try:  # 如果打不开，则下一个
                            image = Image.open(image_path)
                        except FileExistsError:
                            continue

                        if random.random()<=0.2:

                            side = max(bw, bh)
                            cx = int(bx + bw / 2)
                            cy = int(by + bh / 2)
                            image, left, top = crop_center(image, cx, cy, side)
                            if not image:  # 裁剪失败，不够大，就走
                                continue
                            bx = bx - left
                            by = by - top

                            scale = 512 / side
                        else:
                            newimg, scale, bx, by, bw, bh = resize_and_crop(image, 512, bx, by, bw, bh)
                            if not image or min(bw, bh) < 10:
                                continue
                            cx = int(bx + bw / 2)
                            cy = int(by + bh / 2)
                            pos = get_position(cx, cy, 512, 512)
                            size = classify_bbox(bw, bh, 512, 512)

                        image = image.resize((512, 512), Image.LANCZOS)
                        bx = math.floor(scale * bx)
                        by = math.floor(scale * by)
                        bw = math.floor(scale * bw)
                        bh = math.floor(scale * bh)
                        save_path = os.path.join(r"F:\数据集\fmow_fli", '512', category_name)
                        os.makedirs(save_path, exist_ok=True)
                        image.save(os.path.join(save_path, img))
                        dict = {}
                        bounding_box_list['box'][0] = bx
                        bounding_box_list['box'][1] = by
                        bounding_box_list['box'][2] = bw
                        bounding_box_list['box'][3] = bh
                        dict.update({'gsd': gsd / scale})
                        dict.update({'utm': utm})
                        dict.update({'country_code': country})
                        dict.update({'time': time})
                        dict.update({'scan_direction': scan_direction})
                        dict.update({'bounding_box': bounding_box_list})
                        json_file_path = os.path.join(r"F:\数据集\fmow_met", '512', category_name)
                        os.makedirs(json_file_path, exist_ok=True)
                        json_file_path = os.path.join(json_file_path, img.split('.')[0] + '.json')
                        # 将字典写入 JSON 文件
                        with open(json_file_path, "w") as json_file:
                            json.dump(dict, json_file)
                        i += 1
                        iS+=1
                else:


                    try:  # 如果打不开，则下一个
                        image = Image.open(image_path)
                    except FileExistsError:
                        continue
                    if random.random()<0.2:
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                        newimg, left, top = crop_center(image, cx, cy, 512)

                        if newimg==None:
                            continue

                        bx = bx - left
                        by = by - top
                        if bx < 0:
                            bx = 0
                            bw = 512
                        if by < 0:
                            by = 0
                            bh = 512
                    else:
                        newimg, scale, bx, by, bw, bh = resize_and_crop(image, 512, bx, by, bw, bh)
                        if not image or min(bw, bh) < 10:
                            continue
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                        pos = get_position(cx, cy, 512, 512)
                        size = classify_bbox(bw, bh, 512, 512)

                    save_path = os.path.join(r"F:\数据集\fmow_fli", '512', category_name)
                    os.makedirs(save_path, exist_ok=True)
                    newimg.save(os.path.join(save_path, img))
                    dict = {}
                    bounding_box_list['box'][0] = bx
                    bounding_box_list['box'][1] = by
                    bounding_box_list['box'][2] = bw
                    bounding_box_list['box'][3] = bh
                    dict.update({'gsd': gsd})
                    dict.update({'utm': utm})
                    dict.update({'country_code': country})
                    dict.update({'time': time})
                    dict.update({'scan_direction': scan_direction})
                    dict.update({'bounding_box': bounding_box_list})
                    json_file_path = os.path.join(r"F:\数据集\fmow_met", '512', category_name)
                    os.makedirs(json_file_path, exist_ok=True)
                    json_file_path = os.path.join(json_file_path, img.split('.')[0] + '.json')
                    # 将字典写入 JSON 文件
                    with open(json_file_path, "w") as json_file:
                        json.dump(dict, json_file)
                    i += 1
                    iS+=1

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
