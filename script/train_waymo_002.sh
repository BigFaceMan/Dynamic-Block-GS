###
 # @Author: ssp
 # @Date: 2024-10-31 23:27:01
 # @LastEditTime: 2024-11-01 21:57:00
### 
CONFIG_PAHT="/lfs1/users/spsong/Code/Dynamic-Block-GS/configs/example/waymo_train_002_segopt_ninp.yaml"
CUDA_ID=8
CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py --config $CONFIG_PAHT 
CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py --config $CONFIG_PAHT mode evaluate
CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py --config $CONFIG_PAHT 