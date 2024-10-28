'''
Author: ssp
Date: 2024-08-11 19:21:36
LastEditTime: 2024-10-26 17:04:14
'''
import torch 
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        save_dir = os.path.join(cfg.model_path, 'train', "ours_{}_obj_surround".format(scene.loaded_iter))
        os.makedirs(save_dir, exist_ok=True)

        visualizer = Visualizer(save_dir)
        cameras = scene.getVisCameras()
        for vis_obj_name in gaussians.obj_list:
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering {}".format(vis_obj_name))):
                torch.cuda.synchronize()
                result = renderer.render_obj_srd(camera, gaussians, vis_obj_name)
                if result['skip']:
                    continue 
                torch.cuda.synchronize()
                
                visualizer.visualize_obj(result, camera, vis_obj_name)

                
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    render_sets()
