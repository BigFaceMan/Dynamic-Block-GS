'''
Author: ssp
Date: 2024-10-30 11:03:42
LastEditTime: 2024-11-18 10:27:48
'''
import os
from lib.datasets.dataset import sceneLoadTypeCallbacks
from lib.config import cfg
from lib.utils.camera_utils import cameraList_from_camInfos
from lib.datasets.block.data_partition import ProgressiveDataPartitioning

def data_partition():
    """
        scene_info 类型
        scene_info = SceneInfo(
            point_cloud=point_cloud,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=bkgd_ply_path,
            metadata=scene_metadata,
            novel_view_cameras=novel_view_cam_infos,
        )
    """

    scene_info = sceneLoadTypeCallbacks["Waymo"](cfg.source_path, **cfg.data)  # 得到一个场景的所有参数信息
    with open(os.path.join(cfg.model_path, "train_cameras.txt"), "w") as f:
        for cam in scene_info.train_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    with open(os.path.join(cfg.model_path, "test_cameras.txt"), "w") as f:
        for cam in scene_info.test_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    all_cameras = cameraList_from_camInfos(scene_info.train_cameras + scene_info.test_cameras, 1)

    # 开始分块
    DataPartitioning = ProgressiveDataPartitioning(cfg.source_path, scene_info, all_cameras, cfg.model_path,
                                                   cfg.block.m_region, cfg.block.n_region, cfg.block.x_ax, cfg.block.z_ax, cfg.block.extend_rate, cfg.block.visible_rate, cfg.block.data_vis)
    partition_result = DataPartitioning.partition_scene

    # 保存每个partition的图片名称到txt文件
    client = 0
    partition_id_list = []
    for partition in partition_result:
        partition_id_list.append(partition.partition_id)
        camera_info = partition.cameras
        image_name_list = [camera_info[i].camera.image_name for i in range(len(camera_info))]
        txt_file = f"{cfg.model_path}/partition_point_cloud/visible/{partition.partition_id}_camera.txt"
        # 打开一个文件用于写入，如果文件不存在则会被创建
        with open(txt_file, 'w') as file:
            # 遍历列表中的每个元素
            for item in image_name_list:
                # 将每个元素写入文件，每个元素占一行
                file.write(f"{item}\n")
        client += 1

    return client, partition_id_list


def read_camList(path):
    camList = []
    with open(path, "r") as f:
        lines = f.readlines()
        for image_name in lines:
            camList.append(image_name.replace("\n", ""))

    return camList

def get_output():
    print(sceneLoadTypeCallbacks.keys())

if __name__ == '__main__':
    print(sceneLoadTypeCallbacks.keys())