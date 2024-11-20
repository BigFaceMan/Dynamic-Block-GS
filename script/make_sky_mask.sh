###
 # @Author: ssp
 # @Date: 2024-11-20 13:03:22
 # @LastEditTime: 2024-11-20 13:10:45
### 
DATA_DIR="/lfs3/users/spsong/dataset/baseline_dxt/city"
SAM_CKPT="./sam_vit_h_4b8939.pth"
CUDA_VISIBLE_DEVICES=4 python script/waymo/generate_sky_mask.py --datadir $DATA_DIR --sam_checkpoint $SAM_CKPT