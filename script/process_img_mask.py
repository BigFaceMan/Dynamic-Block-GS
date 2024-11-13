'''
Author: ssp
Date: 2024-11-11 12:29:01
LastEditTime: 2024-11-11 12:32:38
'''
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# 定义目录路径
image_dir = '/lfs3/users/spsong/dataset/waymo/training/003/images'  # image 目录
mask_dir = '/lfs3/users/spsong/dataset/waymo/training/003/segment'    # mask 目录
output_dir = '/lfs3/users/spsong/dataset/waymo/training/003/img_after_mask'  # 输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取 image 目录中的所有图像文件
image_files = os.listdir(image_dir)

# 遍历每一张图片
for image_file in tqdm(image_files, total=len(image_files), desc="mask image"):
    # 获取图片路径和对应的掩码路径
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, image_file)  # 假设 mask 和 image 同名
    
    # 检查文件是否存在
    if not os.path.exists(mask_path):
        print(f"Mask file {mask_path} not found for {image_file}. Skipping...")
        continue

    # 打开图像和掩码
    image = Image.open(image_path).convert('RGB')  # 转为 RGB 模式
    mask = Image.open(mask_path).convert('L')  # 转为灰度模式 (L)

    # 将掩码转换为 numpy 数组（0 表示 false，255 表示 true）
    mask_array = np.array(mask)

    # 将图像转换为 numpy 数组
    image_array = np.array(image)

    # 将掩码为 False 的地方替换为白色 (255, 255, 255)
    image_array[mask_array == 0] = [255, 255, 255]

    # 将修改后的数组转换回图像
    output_image = Image.fromarray(image_array)

    # 保存结果
    output_image_path = os.path.join(output_dir, image_file)
    output_image.save(output_image_path)

    # print(f"Processed and saved {image_file}")

print("Processing complete.")
