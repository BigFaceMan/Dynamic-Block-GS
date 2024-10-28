'''
Author: ssp
Date: 2024-08-11 19:21:36
LastEditTime: 2024-10-26 17:15:55
'''
import numpy as np
import os
import shutil
from lib.config import cfg
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model import GaussianModel
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene



if __name__ == '__main__':
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    scene.save_indep(50000)