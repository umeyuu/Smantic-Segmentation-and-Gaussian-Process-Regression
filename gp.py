from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
# import pandas as pd

# 深度推定するピクセルをマスクする関数
def mask(image, xs, xf, ys, yf):
    true = np.empty((xf-xs, yf-ys))
    for x in range(xs, xf):
        for y in range(ys, yf):
            true[x-xs][y-ys] = image[x][y]
            image[x][y] = 0

    return image, true

# 近傍データを見つける関数
def find_neighbor(data, x, y, N):
    
    test = data[x, y, :-1]
    data = data[data[:,:,-1]!=0]
    test = np.array([test for _ in range(data.shape[0])])
    dis = np.expand_dims(np.linalg.norm(data[:,:-1]-test, axis=1), 1)
    df = np.hstack((data, dis))
    df = df[np.argsort(df[:, -1])]

    # x_train = df[:N, :21]
    # y_train = df[:N, -2] 

    x_train = df[:N, :2]
    y_train = df[:N, 2] 
    # breakpoint()

    return x_train, y_train

# 不確実性も考慮して近傍データを見つける関数
def find_neighbor_with_uncer(data, x, y, N):
    test = data[x, y, :-2]
    data = data[data[:,:,-2]!=0]
    test = np.array([test for _ in range(data.shape[0])])
    dis = np.expand_dims(np.linalg.norm(data[:,:-2]-test, axis=1), 1)
    df = np.hstack((data, dis))
    df = df[np.argsort(df[:, -1])]

    x_train = df[:N, :21]
    y_train = df[:N, -3] 
    uncer = df[:N, -2]

    return x_train, y_train, uncer


def calc_kernel_matrix(X, kernel):
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j])

    return K


def calc_kernel_matrix_with_uncer(X, uncer, kernel):
    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # if uncer[i] > uncer[j]:
            #     K[i, j] = kernel(X[i], X[j]) - uncer[i]/10
            # else:
            #     K[i, j] = kernel(X[i], X[j]) - uncer[j]/10
            K[i, j] = kernel(X[i], X[j]) + 0.5*(uncer[j] * uncer[i])

    return K

def calc_kernel_sequence(X, x, kernel):
    N = len(X)
    seq = np.zeros((N))
    for i in range(N):
        seq[i] = kernel(X[i], x)
        
    return seq

def calc_kernel_sequence_with_uncer(X, x, Uncer, uncer, kernel):
    N = len(X)
    seq = np.zeros((N))
    for i in range(N):
        # if Uncer > uncer[i]:
        #     seq[i] = kernel(X[i], x) - Uncer/10
        # else:
        #     seq[i] = kernel(X[i], x) - uncer[i]/10
        seq[i] = kernel(X[i], x) #+ 0.5*(uncer[i] * Uncer)
        
    return seq

# ガウス過程回帰で予測を行う関数
def predict(X_train, Y_train, X, kernel, sigma2_y):
    N = len(X)
    mu = np.zeros((N))
    sigma2 = np.zeros(N)
    N_train = len(Y_train)
    K = calc_kernel_matrix(X_train, kernel)
    # breakpoint()
    invmat = np.linalg.inv(sigma2_y*np.eye(N_train) + K)
    
    seq = calc_kernel_sequence(X_train, X, kernel)
    mu = (seq.reshape(1, N_train) @ invmat) @ Y_train
    sigma2 = np.sqrt(sigma2_y + kernel(X, X) - (seq.reshape(1, N_train) @ invmat) @ seq.reshape(N_train, 1))

    return mu, sigma2

# 説明変数の不確実性を考慮したガウス過程回帰で予測を行う関数
def predict_with_uncer(X_train, Y_train, X, uncer, Uncer, kernel, sigma2_y):
    N = len(X)
    mu = np.zeros((N))
    sigma2 = np.zeros(N)
    N_train = len(Y_train)
    K = calc_kernel_matrix_with_uncer(X_train, uncer, kernel)
    invmat = np.linalg.inv(sigma2_y*np.eye(N_train) + K)
    
    seq = calc_kernel_sequence_with_uncer(X_train, X, Uncer, uncer, kernel)
    mu = (seq.reshape(1, N_train) @ invmat) @ Y_train
    sigma2 = np.sqrt(sigma2_y + kernel(X, X) - (seq.reshape(1, N_train) @ invmat) @ seq.reshape(N_train, 1))
        
        
    return mu, sigma2

# 説明変数の不確実性を考慮したガウス過程回帰を実行する
def exe_with_uncer(path, xs, xf, ys, yf):
    # RBF covariance function
    alpha = 1.2
    beta = 0.5
    kernel_g = lambda x1, x2: alpha * np.exp(-0.5*(beta**2)*np.linalg.norm(x1 - x2))

    sigma2_y = 1.0
    data_dir = path

    data = np.load(data_dir + 'data.npy')
    uncer = np.expand_dims(np.load(data_dir + 'uncertainty.npy'), 0)
    data = np.concatenate([data, uncer]).transpose(1, 2, 0)

    data[:, :, -2], true = mask(data[:,:,-2], xs, xf, ys, yf)

    mask_data = data[:, :, -2].copy()

    pred = true.copy()
    var = true.copy()

    for i, x in enumerate(tqdm(range(xs, xf))):
        for j, y in enumerate(range(ys, yf)):
            x_train, y_train, uncer_train = find_neighbor_with_uncer(data, x, y, 50)
            x_test = data[x, y, :-2].reshape(1,-1)
            Uncer = uncer[0, x, y]
            mu_g, sigma2_g = predict_with_uncer(x_train, y_train, x_test, uncer_train, Uncer, kernel_g, sigma2_y)
            mask_data[x, y] = mu_g
            pred[i, j] = mu_g
            var[i, j] = sigma2_g

    # data_dir = f'sample_result/data{index}/'
    # if data_dir is not None:
    #             os.makedirs(data_dir, exist_ok=True)

    np.save(data_dir + 'pred', pred)
    np.save(data_dir + 'var', var)

    plt.figure(figsize=(20,10))
    plt.imshow(mask_data ,interpolation='nearest',cmap='jet')
    #plt.imshow(true ,interpolation='nearest',cmap='jet')
    plt.colorbar()
    plt.savefig(data_dir + 'depth.png')


# ガウス過程回帰を実行する関数
def exe(path, xs, xf, ys, yf, index):
    # RBF covariance function
    alpha = 1.2
    beta = 0.5
    kernel_g = lambda x1, x2: alpha * np.exp(-0.5*(beta**2)*np.linalg.norm(x1 - x2))
    #K_g = calc_kernel_matrix(x_train, kernel_g)

    sigma2_y = 1.0

    data_dir = path
    #args[2:] = [int(arg) for arg in args[2:]]

    data = np.load(data_dir + 'data.npy')#.transpose(1, 2, 0)
    xy = data[:2]
    data = np.concatenate([xy, [data[-1]]]).transpose(1, 2, 0)

    data[:, :, -1], true = mask(data[:,:,-1], xs, xf, ys, yf)

    mask_data = data[:, :, -1].copy()

    pred = true.copy()
    var = true.copy()

    for i, x in enumerate(tqdm(range(xs, xf))):
        for j, y in enumerate(range(ys, yf)):
            x_train, y_train = find_neighbor(data, x, y, 50)
            x_test = data[x, y, :-1].reshape(1,-1)
            mu_g, sigma2_g = predict(x_train, y_train, x_test, kernel_g, sigma2_y)
            mask_data[x, y] = mu_g
            pred[i, j] = mu_g
            var[i, j] = sigma2_g
    
    data_dir = f'nomal_result/data{index}/'
    if data_dir is not None:
                os.makedirs(data_dir, exist_ok=True)
    np.save(data_dir + 'pred', pred)
    np.save(data_dir + 'var', var)
    

    plt.figure(figsize=(20,10))
    plt.imshow(mask_data ,interpolation='nearest',cmap='jet')
    #plt.imshow(true ,interpolation='nearest',cmap='jet')
    plt.colorbar()
    plt.savefig(data_dir + 'depth.png')
