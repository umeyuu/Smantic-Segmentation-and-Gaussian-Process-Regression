# Prediction of Uncertain Values in Depth Images by Segmentation and Gaussian Process Regression Leading to Its Generalization.

We propose a method for predicting the depth value of missing parts of depth images using Gaussian process regression. We extracted the category likelihood of pixels in RGB images using semantic segmentation, and estimated the depth using Gaussian process regression on the feature space with the location information added. To account for uncertainty in the category likelihood, a penalty term was introduced into the covariance matrix. By generalizing the proposed method, we were able to derive a Gaussian process regression that takes into account the observed noise of the explanatory variables. This generalized method does not treat all input data equally, but seems to perform regression with a preference based on the uncertainty of the input data, leading to the conclusion that it is a highly versatile method.


https://github.com/umeyuu/Smantic-Segmentation-and-Gaussian-Process-Regression/issues/1#issue-1429544586



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
## 3. Predict semantic segmentation
```
python predict.py --input data_path  --dataset cityscapes --model deeplabv3plus_resnet101 --ckpt model_path --save_val_results_to save_dir
```

## 4. Estimating depth per pixel with Gaussian process regression using semantic segmentation results
```
python GP_exe_evaluate.py
```
