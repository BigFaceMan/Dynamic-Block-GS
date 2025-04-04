###
 # @Author: ssp
 # @Date: 2024-11-20 16:59:32
 # @LastEditTime: 2024-11-22 10:23:13
### 
CONFIG_PATH="/lfs1/users/spsong/Code/Dynamic-Block-GS/configs/example/test.yaml"
CUDA_ID=3

# 运行训练、渲染和度量脚本
CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py --config $CONFIG_PATH
CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py --config $CONFIG_PATH mode evaluate
CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py --config $CONFIG_PATH
