import cv2
import torch
import numpy as np
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 配置Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# COCO中的汽车类别ID为2（从0开始计数）
CAR_CLASS_ID = 2

def segment_cars(image_path):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    instances = outputs["instances"]
    pred_classes = instances.pred_classes
    pred_masks = instances.pred_masks

    car_masks = []
    for cls, mask in zip(pred_classes, pred_masks):
        if cls == CAR_CLASS_ID:
            car_masks.append(mask.cpu().numpy())

    if not car_masks:
        return None  # 没有检测到车辆

    # 合并所有车辆的mask
    combined_mask = np.zeros_like(car_masks[0], dtype=np.uint8)
    for mask in car_masks:
        combined_mask = np.logical_or(combined_mask, mask)
    combined_mask = combined_mask.astype(np.uint8) * 255

    return combined_mask

# 处理所有图片
input_folder = '/lfs3/users/spsong/dataset/waymo/training/003/images'
output_folder = '/lfs3/users/spsong/dataset/waymo/training/003/segment'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        mask = segment_cars(img_path)
        if mask is not None:
            cv2.imwrite(os.path.join(output_folder, filename), mask)
        else:
            print(f"No cars detected in {filename}")

