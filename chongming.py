import os

import os

def delete_images_in_folder(folder_path, image_extensions=['.jpg', '.jpeg', '.png']):
    """
    删除指定文件夹及其子文件夹中的所有图像文件
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"已删除文件：{file_path}")

# 指定要删除图像的文件夹路径
folder_path = r'F:\数据集\fmow_flit'

# 调用函数删除图像文件
delete_images_in_folder(folder_path)