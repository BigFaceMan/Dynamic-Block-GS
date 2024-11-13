<!--
 * @Author: ssp
 * @Date: 2024-10-31 23:34:34
 * @LastEditTime: 2024-11-11 12:28:02
-->
# Dynamic-Block-GS

## 数据集 
dynamic_mask 用于表明物体的位置，可以放bbox也可以放seg，用于后面的seg损失
mask 是全局的mask loss时被加入损失函数里，用于只计算部分的损失

### 如何生成seg
conda activate detectron 

更改script/make_car_seg.py中的scr 和 tag路径

python script/make_car_seg.py


