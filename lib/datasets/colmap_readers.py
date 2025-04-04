import os
import sys
import cv2
import numpy as np
from PIL import Image
from lib.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from lib.utils.colmap_utils import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from lib.config import cfg
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly, get_Sphere_Norm

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, scene_info_waymo=None):
    cam_infos = []
    sky_mask_folder = os.path.join(images_folder[:-len(os.path.basename(images_folder))], 'sky_mask')
    have_sky_mask = os.path.exists(sky_mask_folder)
    print("Colmap sky_mask_folder : ", sky_mask_folder)
    p_log = True
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == "SIMPLE_PINHOLE":
            focal_length = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length, height)
            FovX = focal2fov(focal_length, width)
            K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]]).astype(np.float32)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1]]).astype(np.float32)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([[focal_length_x, 0, cx], [0, focal_length_x, cy], [0, 0, 1]]).astype(np.float32)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]  # 主点横坐标
            cy = intr.params[3]  # 主点纵坐标
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1]]).astype(np.float32)
        else:
            print("Camera model is: ", intr.model)
            assert False, "COLMAP camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        image = Image.open(image_path)
        try:
            # images
            mask_path = os.path.join(os.path.dirname(images_folder), "mask", extr.name)
            # print("\nmask_path : ", mask_path)
            # print("mask path is : ", mask_path)
            img_mask = Image.open(mask_path).convert("L")  
            # print("mask shape is : ", np.array(mask).shape)
        except:
            img_mask = None
        
        if scene_info_waymo != None and cfg.train.waymo_pose:
            if p_log:
                print("<< use waymo pose >>")
            for idx in range(len(scene_info_waymo.train_cameras)):
                img1_name = image_name
                img2_name = scene_info_waymo.train_cameras[idx].image_name
                img1_name_m = img1_name.split('.')[0].split('/')[1] + '_' + img1_name.split('/')[0].split('_')[1]
                # print("img1  : ", img1_name_m)
                # print("img2  : ", img2_name)
                if img1_name_m == img2_name:
                    R = scene_info_waymo.train_cameras[idx].R
                    T = scene_info_waymo.train_cameras[idx].T
                    K = scene_info_waymo.train_cameras[idx].K
                    FovY = scene_info_waymo.train_cameras[idx].FovY
                    FovX = scene_info_waymo.train_cameras[idx].FovX
                    metadata_waymo = scene_info_waymo.train_cameras[idx].metadata
                    # print("chang {} cam info".format(img1_name))
                    break
        
        metadata = {}

        # read sky_mask
        if have_sky_mask:
            sky_mask_path = os.path.join(sky_mask_folder, extr.name)
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            sky_mask = Image.fromarray(sky_mask)
            metadata['sky_mask'] = sky_mask

        # 第几个相机，为后面优化天空做准备
        metadata['cam'] = 0
        if scene_info_waymo != None and cfg.train.waymo_meta:
            if p_log:
                print("<< use waymo metadata >>")
            metadata = metadata_waymo

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, K=K, 
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height, metadata=metadata, mask=img_mask)
        cam_infos.append(cam_info)
        p_log = False
    sys.stdout.write('\n')

    return cam_infos


def readColmapSceneInfo(path, scene_info_waymo: SceneInfo=None, images='images', split_test=8, **kwargs):
    colmap_basedir = os.path.join(path, 'sparse/0')
    if not os.path.exists(colmap_basedir):
        colmap_basedir = os.path.join(path, 'sparse')
    try:
        cameras_extrinsic_file = os.path.join(colmap_basedir, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_basedir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_basedir, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_basedir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), scene_info_waymo=scene_info_waymo)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if split_test == -1:
        train_cam_infos = cam_infos
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % split_test != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % split_test == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(colmap_basedir, "points3D.ply")
    bin_path = os.path.join(colmap_basedir, "points3D.bin")
    txt_path = os.path.join(colmap_basedir, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None 
    if scene_info_waymo != None and cfg.train.waymo_point:
        print("<< this colmap train use waymo pcd >>")
        pcd = scene_info_waymo.point_cloud
    # if scene_info_waymo != None and cfg.train.waymo_pose:
    #     print("<< this colmap train use waymo pose >>")
    #     train_cam_infos = scene_info_waymo.train_cameras
        

    scene_metadata = dict()

    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    sphere_normalization = get_Sphere_Norm(pcd.points)
    scene_metadata['sphere_center'] = sphere_normalization['center']
    scene_metadata['sphere_radius'] = sphere_normalization['radius']

    print("before Colmap scene_radius : ", scene_metadata['scene_radius'])
    print("before Colmap scene_center : ", scene_metadata['scene_center'])
    print("before Colmap sphere_radius : ", scene_metadata['sphere_radius'])
    print("before Colmap shpere_center : ", scene_metadata['sphere_center'])

    if scene_info_waymo != None and cfg.train.waymo_smeta:
        # print("<< this colmap train use waymo smeta >>")
        print("<< this colmap train use waymo scene_center >>")
        scene_metadata["scene_radius"] = scene_info_waymo.metadata["scene_radius"]

        print("after Colmap scene_radius : ", scene_metadata['scene_radius'])
        print("after Colmap scene_center : ", scene_metadata['scene_center'])
        print("after Colmap sphere_radius : ", scene_metadata['sphere_radius'])
        print("after Colmap shpere_center : ", scene_metadata['sphere_center'])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           metadata=scene_metadata)
    return scene_info