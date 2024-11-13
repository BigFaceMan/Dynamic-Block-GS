'''
Author: ssp
Date: 2024-11-09 16:26:37
LastEditTime: 2024-11-11 12:35:17
'''
import numpy
from PIL import Image

def crop_image(input_path, output_path, top_left_x, top_left_y, size=700):
    # 打开图像
    img = Image.open(input_path)
    
    # 定义裁剪区域 (左, 上, 右, 下)
    crop_area = (top_left_x, top_left_y, top_left_x + size, top_left_y + size)
    
    # 裁剪图像
    cropped_img = img.crop(crop_area)
    
    # 保存裁剪后的图像
    cropped_img.save(output_path)
    print(f"Cropped image saved as {output_path}")

# 使用示例
input_image_path = '/lfs3/users/spsong/dataset/waymo/training/003/img_after_mask/000000_0.png'   # 输入图片路径
output_image_path = './crop_output.png' # 输出图片路径
top_left_x = 736                 # 裁剪区域左上角的 x 坐标
top_left_y = 518                # 裁剪区域左上角的 y 坐标



crop_image(input_image_path, output_image_path, top_left_x, top_left_y)
