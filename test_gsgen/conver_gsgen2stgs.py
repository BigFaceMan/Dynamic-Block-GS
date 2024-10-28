'''
Author: ssp
Date: 2024-10-24 18:50:03
LastEditTime: 2024-10-25 23:02:36
'''
import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser, Namespace

# path = "./output/waymo_full_exp/waymo_train_002_0_198/point_cloud/iteration_50000_all/point_cloud_obj_004.ply"

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def convert(input_path, output_path):
    # [x, y, z, opacity, f_dc_, f_rest_, scale_, rot_]
    # "x",  "y",  "z",  "nx",  "ny",  "nz",  "red",  "green",  "blue",  "opacity",  "scale_0",  "scale_1",  "scale_2",  "rot_0",  "rot_1",  "rot_2",  "rot_3",
    plydata = PlyData.read(input_path)
    plydata = plydata.elements[0]
    xyz = np.stack((np.asarray(plydata["x"]),
                    np.asarray(plydata["y"]),
                    np.asarray(plydata["z"])),  axis=1)
    normals = np.zeros_like(xyz)
    opacities = np.asarray(plydata["opacity"])[..., np.newaxis]

    rgb_f_names = ["red", "green", "blue"]
    rgb_dc = np.zeros((xyz.shape[0], len(rgb_f_names)))
    for idx, attr_name in enumerate(rgb_f_names):
        rgb_dc[:, idx] = np.asarray(plydata[attr_name]) / 255.0
    print(rgb_dc.shape)
    features_dc = RGB2SH(rgb_dc)
    features_dc = rgb_dc
    features_rest = np.zeros((xyz.shape[0], 9))
    # features_dc = features_dc.reshape(features_dc.shape[0], 3, -1)
    # features_rest = features_rest.reshape(features_rest.shape[0], 3, -1)

    scale_names = [p.name for p in plydata.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata[attr_name])

    rot_names = [p.name for p in plydata.properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata[attr_name])


    semantic = np.zeros((xyz.shape[0], 0))

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rots.shape[1]):
        l.append('rot_{}'.format(i))
    for i in range(semantic.shape[1]):
        l.append('semantic_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]
    print("dtype_full : ", dtype_full)

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, features_dc, features_rest, opacities, scales, rots, semantic), axis=1)
    # attributes = np.concatenate((xyz, normals, features_dc, opacities, scales, rots, semantic), axis=1)
    elements[:] = list(map(tuple, attributes))
    elements = PlyElement.describe(elements, 'vertex')
    PlyData([elements]).write(output_path)
    print("Convert OK !!!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--input_path", type=str, default="./gsgen_ply/a_white_car.ply")
    parser.add_argument("--output_path", type=str, default="./gsgen_ply/point_cloud_obj_004.ply")
    args = parser.parse_args(sys.argv[1:])
    convert(args.input_path, args.output_path)