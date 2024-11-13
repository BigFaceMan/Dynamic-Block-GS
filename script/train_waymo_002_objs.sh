###
 # @Author: ssp
 # @Date: 2024-11-05 16:28:36
 # @LastEditTime: 2024-11-13 15:55:55
### 
CONFIG_PATH="/lfs1/users/spsong/Code/Dynamic-Block-GS/configs/example/waymo_train_002_objs_s0100reg.yaml"
CUDA_ID=5

# 运行训练、渲染和度量脚本
CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py --config $CONFIG_PATH
CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py --config $CONFIG_PATH mode trajectory
CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py --config $CONFIG_PATH
