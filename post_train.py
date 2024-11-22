import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    gaussians.training_setup()
    scene = Scene(gaussians=gaussians, dataset=dataset, post_train_stage=True)

    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass
    # only optimizer opacity
    gaussians.post_training_setup();

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    post_train_len = 1000
    progress_bar = tqdm(range(start_iter, training_args.iterations + post_train_len))
    start_iter += 1

    viewpoint_stack = None
    output_loss_info = True
    for iteration in range(start_iter, training_args.iterations + post_train_len + 1):
    
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    
        # ====================================================================
        # Get mask
        # original_mask: pixel in original_mask with 0 will not be surpervised
        # original_acc_mask: use to suepervise the acc result of rendering
        # original_sky_mask: sky mask

        gt_image = viewpoint_cam.original_image.cuda()
        if hasattr(viewpoint_cam, 'original_mask'):
            mask = viewpoint_cam.original_mask.cuda().bool()
            if output_loss_info:
                print("use original_mask")
        else:
            mask = torch.ones_like(gt_image[0:1]).bool()
            if output_loss_info:
                print("use generate_ones_mask")
        
        if hasattr(viewpoint_cam, 'original_sky_mask'):
            sky_mask = viewpoint_cam.original_sky_mask.cuda()
            if output_loss_info:
                print("use original_sky_mask")
        else:
            sky_mask = None
            if output_loss_info:
                print("no sky_mask")
            
        if hasattr(viewpoint_cam, 'original_obj_bound'):
            obj_bound = viewpoint_cam.original_obj_bound.cuda().bool()
            if output_loss_info:
                print("use original_obj_bound")
        else:
            obj_bound = torch.zeros_like(gt_image[0:1]).bool()
            if output_loss_info:
                print("use generates_ones_objmask")
        
        if (iteration - 1) == training_args.debug_from:
            cfg.render.debug = True
            
        
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)

        # skip no data case
        if render_pkg["skip"] or torch.sum(mask).item() == 0:
            continue ;
        
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth'] # [1, H, W]

        scalar_dict = dict()
        # rgb loss
        # Ll1 = l1_loss(image, gt_image, mask)
        # scalar_dict['l1_loss'] = Ll1.item()
        # loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
        # bkground reg
        # 会把很多点给删除掉
        # 只用背景的去做loss
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            # render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
            # image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc, min=1e-6, max=1.-1e-6)

            # 将 obj_bound 为 0 的位置设置为 -torch.log(1. - acc_obj)
            # 只回传给背景位置的点
            obj_acc_loss = torch.where(obj_bound, torch.tensor(0.0, device=obj_bound.device), -torch.log(1. - acc_obj))
            # 只计算 obj_bound 为 0 的位置的均值
            obj_acc_loss = obj_acc_loss[~obj_bound].mean()
            # obj_acc_loss = torch.where(obj_bound, 
            #     -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
            #     -torch.log(1. - acc_obj)).mean()
            loss = obj_acc_loss

            if output_loss_info:
                print("use obj_loss")


        loss.backward()
        
        iter_end.record()
                
        is_save_images = True
        output_loss_info = False
        with torch.no_grad():

            if iteration % 10 == 0:
                progress_bar.update(10)
            if iteration == training_args.iterations:
                progress_bar.close()
            gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
            gaussians.parse_camera(viewpoint_cam)   
            gaussians.set_max_radii2D(radii, visibility_filter)
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration % optim_args.prune_min_opacity_interval == 0:
                scalars, tensors = gaussians.prune_min_opacity(
                    min_opacity=optim_args.post_min_opacity
                )


            # Optimizer step
            gaussians.post_updat_optimizer()

            if (iteration == training_args.iterations + post_train_len):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=False)
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)



def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)
    # print(torch.cuda.current_device())  # 打印当前设备编号
    # print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 打印当前设备名称

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)


    # Start GUI server, configure and run training
    # torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")