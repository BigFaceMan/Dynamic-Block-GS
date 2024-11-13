'''
Author: ssp
Date: 2024-11-09 21:25:05
LastEditTime: 2024-11-09 21:27:33
'''
import numpy as np
from PIL import Image

img_path = '/lfs3/users/spsong/dataset/waymo/training/002/img_after_mask/000000_0.png'
mask_path = '/lfs3/users/spsong/dataset/waymo/training/002/mask/000000_0.png'
output_path = '/lfs3/users/spsong/dataset/waymo/training/002/rgba_mask/000000_0.png'
# 加载 RGB 图片和 mask 图片
rgb_image = Image.open(img_path).convert("RGB")
mask_image = Image.open(mask_path).convert("L")  # 灰度模式

# 将图像转换为 numpy 数组
rgb_array = np.array(rgb_image)
mask_array = np.array(mask_image)

# 创建一个全白的背景 (255, 255, 255, 0) 的 RGBA 图层
rgba_array = np.ones((rgb_array.shape[0], rgb_array.shape[1], 4), dtype=np.uint8) * 255
rgba_array[..., 3] = 0  # 将 alpha 通道设为 0

# 根据 mask 设置 alpha 通道和 RGB 值
# 假设 mask 中非零的部分是需要保留的部分
mask_indices = mask_array > 0

# 设置 mask 区域：保持原始 RGB 值并设定 alpha 为 255
rgba_array[mask_indices, :3] = rgb_array[mask_indices]
rgba_array[mask_indices, 3] = 255

# 将 numpy 数组转换回 PIL 图像
rgba_image = Image.fromarray(rgba_array, "RGBA")

# 保存结果
rgba_image.save(output_path)
