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
data_root_dir = r"F:\数据集\fmow\val"

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

    if right>w or bottom > h or left<0 or top<0:#超了
        return None,None,None

    newimg=img.crop((left,top,right,bottom))




    return newimg,left,top

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


            img=img[:-4]+'_val'+img[-4:]

            if min(w,h)>=1024:
                if max(bw,bh)>=1024: #其中一个大于1024
                    if max(bw,bh)>4096: #其中一个大于4096
                        continue
                    else:
                        try:  # 如果打不开，则下一个
                            image = Image.open(image_path)
                        except FileExistsError:
                            continue

                        side=max(bw,bh)
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                        image,left,top=crop_center(image,cx,cy,side)
                        if not image: #裁剪失败，不够大，就走
                            continue
                        bx = bx - left
                        by = by - top

                        scale= 1024/side

                        image= image.resize((1024, 1024), Image.LANCZOS)
                        bx=math.floor(scale*bx)
                        by=math.floor(scale*by)
                        bw=math.floor(scale*bw)
                        bh= math.floor (scale*bh)
                        save_path=os.path.join( r"F:\数据集\fmow_fli",'1024',category_name)
                        os.makedirs(save_path, exist_ok=True)
                        image.save(os.path.join(save_path,img))
                        dict={}
                        bounding_box_list['box'][0]=bx
                        bounding_box_list['box'][1]=by
                        bounding_box_list['box'][2]=bw
                        bounding_box_list['box'][3]=bh
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
                    cx=int(bx+bw/2)
                    cy=int(by+bh/2)
                    newimg,left,top=crop_center(image,cx,cy,1024)

                    if newimg==None:
                        continue

                    bx=bx-left
                    by=by-top
                    if bx<0:
                        bx=0
                        bw=1024
                    if by<0:
                        by=0
                        bh=1024
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

                        side = max(bw, bh)
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                        image, left, top = crop_center(image, cx, cy, side)
                        if not image:  # 裁剪失败，不够大，就走
                            continue
                        bx = bx - left
                        by = by - top

                        scale = 512 / side

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
