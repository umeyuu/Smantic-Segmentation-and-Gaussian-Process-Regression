# Prediction of Uncertain Values in Depth Images by Segmentation and Gaussian Process Regression Leading to Its Generalization.

We propose a method for predicting the depth value of missing parts of depth images using Gaussian process regression. We extracted the category likelihood of pixels in RGB images using semantic segmentation, and estimated the depth using Gaussian process regression on the feature space with the location information added. To account for uncertainty in the category likelihood, a penalty term was introduced into the covariance matrix. By generalizing the proposed method, we were able to derive a Gaussian process regression that takes into account the observed noise of the explanatory variables. This generalized method does not treat all input data equally, but seems to perform regression with a preference based on the uncertainty of the input data, leading to the conclusion that it is a highly versatile method.


|Process flow of the proposed method|Depth estimation of preceding and proposed method 1.|
|---|:---:|
|![image](https://user-images.githubusercontent.com/91179464/198981993-40903477-38e1-4888-8f90-4fcd2f2b3a7b.png)|![image](https://user-images.githubusercontent.com/91179464/198981159-9dea102f-f99c-4cca-be3d-c43dc36a9ab6.png)|

## Proposed Gaussian Regression

We propose to introduce a penalty term in the covariance matrix of a Gaussian process when there is uncertainty in the input. Let u_x be the uncertainty of the input x. The formulation is as follows.

█(f ~ N(k_*^T K^(-1) Y, k_(**)-k_*^T K^(-1) k_* )#(1) )
█(k_*=(k(x_*, x_1 )… k(x_*, x_N ))#(2) )
█(k_(**)=k(x_*, x_* )#(3) )
█(K=(■(k(x_1, x_1 )+w(u_(x_1 )×u_(x_1 ) )&⋯&k(z_1, z_N )+w(u_(x_1 )×u_(x_N ) )@⋮&⋱&⋮@k(x_N, x_1 )+w(u_(x_N )×u_(x_1 ) )&⋯&k(x_N, x_N )+w(u_(x_N )×u_(x_N ) ) ))#(4) )

where w is the weight parameter and k(x,x^' ) is the kernel function. 

***


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
