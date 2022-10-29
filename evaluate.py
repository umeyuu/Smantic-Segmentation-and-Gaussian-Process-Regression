import numpy as np
import matplotlib.pyplot as plt
import sys


def find_neighbor(data, x, y, N):
    distance = np.empty((384, 480))
    dis_xy = []

    for i, row in enumerate(data):
        for j, d in enumerate(row):
            dis = d[:-1] - data[x][y][:-1]
            dis = np.linalg.norm(dis, ord=2)
            distance[i][j] = dis
            dis_xy.append(dis)

    return distance

def visualize(data_dir, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image ,interpolation='nearest',cmap='jet')
        plt.colorbar()
    plt.savefig(data_dir + 'error.png', format="png")

def mask(image, xs, xf, ys, yf):
    true = np.empty((xf-xs, yf-ys))
    for x in range(xs, xf):
        for y in range(ys, yf):
            true[x-xs][y-ys] = image[x][y]
            image[x][y] = 0

    return image, true

# 深度推定の精度を評価する関数
def exe(path, path1, xs, xf, ys, yf):
    data_dir = path
    #args[2:] = [int(arg) for arg in args[2:]]
    data = np.load(data_dir + 'data.npy').transpose(1, 2, 0)
    data[:, :, -1], true = mask(data[:,:,-1], xs, xf, ys, yf)

    data_dir = path1
    pred = np.load(data_dir + 'pred.npy')
    var = np.load(data_dir + 'var.npy')
    # uncer = np.load(data_dir + 'uncertainty.npy')
    a = true.copy()
    score = 0
    i=0
    for x in range(true.shape[0]):
        for y in range(true.shape[1]):
            if true[x, y]==0:
                a[x, y] = 0
            else:
                i = i +1
                a[x, y] = np.abs((pred[x, y]- true[x, y])/true[x, y])
                score = score + np.abs((pred[x, y]- true[x, y])/true[x, y])
    f = open('test_results/score.txt', 'a')
    if i != 0:
        f.write('{}, {}\n'.format(score/i, i))
    else:
        f.write('nan\n')
    f.close()
    
    # print(score/i)
    # plt.figure(figsize=(20,10))
    # plt.imshow(a ,interpolation='nearest',cmap='jet')
    # plt.colorbar()
    # plt.savefig(data_dir + '/distance.png')

    visualize(
        data_dir, 
        prediction = pred,
        variance = var,
        true = true,
        error = a,
        # uncer = uncer[xs:xf, ys:yf]
    )

    return a, var

# t検定
def t_test(path, path1, xs, xf, ys, yf):
    data_dir = path
    #args[2:] = [int(arg) for arg in args[2:]]
    data = np.load(data_dir + 'data.npy').transpose(1, 2, 0)
    data[:, :, -1], true = mask(data[:,:,-1], xs, xf, ys, yf)

    data_dir = path1
    pred = np.load(data_dir + 'pred.npy')
    var = np.load(data_dir + 'var.npy')
    # uncer = np.load(data_dir + 'uncertainty.npy')
    a = true.copy()
    score = 0
    i=0
    for x in range(true.shape[0]):
        for y in range(true.shape[1]):
            if true[x, y]==0:
                a[x, y] = 0
            else:
                i = i +1
                a[x, y] = np.abs((pred[x, y]- true[x, y])/true[x, y])
                score = score + np.abs((pred[x, y]- true[x, y])/true[x, y])

    # visualize(
    #     data_dir, 
    #     prediction = pred,
    #     variance = var,
    #     true = true,
    #     error = a,
    #     # uncer = uncer[xs:xf, ys:yf]
    # )
    
    a = np.array(a.reshape(1,-1)[0])
    var = np.array(var.reshape(1,-1)[0])
    results = []

    for e, v in zip(a, var):
        if e != 0 and e<=3:
            results.append([e, v])
            
    results = np.array(results).T
    mean = results[0].mean()
    corr = np.corrcoef(results)[0][1]

    return mean, corr
