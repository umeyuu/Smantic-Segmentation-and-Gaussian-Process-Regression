# Smantic-Segmentation-and-Gaussian-Process-Regression

## 1. Download Citysoace and place it as follow
```
/datasets
    /data
        /gtFine
        /leftImg8bit
```
## 2. Train your model on Cityscapes
```
python train.py --model deeplabv3plus_mobilenet --dataset cityscapes --enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/cityscapes 
```
