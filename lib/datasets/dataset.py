'''
Author: ssp
Date: 2024-10-23 21:14:25
LastEditTime: 2024-11-20 20:11:52
'''
import os
import random
import json
from lib.utils.camera_utils import camera_to_JSON, cameraList_from_camInfos, cameraList_from_camInfos_vis_obj
from lib.config import cfg
from lib.datasets.base_readers import storePly, SceneInfo
from lib.datasets.colmap_readers import readColmapSceneInfo
from lib.datasets.blender_readers import readNerfSyntheticInfo
from lib.datasets.waymo_full_readers import readWaymoFullInfo, partitonReadWaymoFullInfo

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Waymo": readWaymoFullInfo,
    "PartitonWaymo": partitonReadWaymoFullInfo,
}

class Dataset():
    def __init__(self):
        self.cfg = cfg.data
        self.model_path = cfg.model_path
        self.source_path = cfg.source_path
        self.images = self.cfg.images

        self.train_cameras = {}
        self.test_cameras = {}
        self.vis_obj_cameras = {}

        dataset_type = cfg.data.get('type', "Colmap")
        if dataset_type == "Colmap_test":
            dataset_type = "Waymo"
            scene_info_waymo: SceneInfo = sceneLoadTypeCallbacks[dataset_type]("/lfs3/users/spsong/dataset/waymo/training/002", **cfg.data)
            dataset_type = "Colmap"
            scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, scene_info_waymo=scene_info_waymo, **cfg.data)
        else:
            assert dataset_type in sceneLoadTypeCallbacks.keys(), 'Could not recognize scene type!'
            scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, **cfg.data)

        if cfg.mode == 'train':
            print(f'Saving input pointcloud to {os.path.join(self.model_path, "input.ply")}')
            pcd = scene_info.point_cloud
            storePly(os.path.join(self.model_path, "input.ply"), pcd.points, pcd.colors)

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))

            print(f'Saving input camera to {os.path.join(self.model_path, "cameras.json")}')
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
       
        self.scene_info = scene_info
        
        if self.cfg.shuffle and cfg.mode == 'train':
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        for resolution_scale in cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale)
            self.vis_obj_cameras[resolution_scale] = cameraList_from_camInfos_vis_obj(self.scene_info.train_cameras, resolution_scale)
            