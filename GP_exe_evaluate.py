# import model
import gp
import evaluate
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    
    #model.exe()

    xmax = 352
    ymax = 1216
    a = []
    b = []

    # #preは２２５までおｋ
    # i=216
    # path = 'pre_test_results/data{}/'.format(i)
    # path1 = 'nomal_result/data216/'

    # xs = 180
    # xf = 220
    # ys = 450
    # yf = 600
    
    # error, var = evaluate.exe(path, path1, xs, xf, ys, yf)
    # error = error.reshape(1,-1)[0]
    # for e in error:
    #     if e>0:
    #         a.append(e)
    # breakpoint()

    # empty = [235, 250, 268, 274, 288, 291]

    for i in tqdm(range(0, 300)):
        # path = 'pre_test_results/data{}/'.format(i)
        # path1 = 'sample_result/data{}/'.format(i)
        # path1 = 'nomal_result/data{}/'.format(i)
        # path = 'test_results/data{}/'.format(i)
        path = 'penalty_only_test_results/data{}/'.format(i)

        # マスクする領域をランダムに決める
        # xs = random.randint(100,xmax)
        # xf = xs + 80
        # ys = random.randint(0,ymax)
        # yf = ys + 30
        # if random.randint(0,1)==1:
        #     xf = xs + 30
        #     yf = ys + 80
        # if xf >= xmax:
        #     tmp = xf - xs
        #     xf = xs
        #     xs = xf - tmp
        # if yf >= ymax:
        #     tmp = yf - ys
        #     yf = ys
        #     ys = yf - tmp

        # マスクする領域を固定する
        xs = 180
        xf = 220
        ys = 450
        yf = 600
        
        
        try:
            # 説明変数の不確実性を考慮したガウス過程回帰を実行
            # gp.exe_with_uncer(path, xs, xf, ys, yf)

            # ガウス過程回帰を実行する
            gp.exe(path, xs, xf, ys, yf, i)

            # 予測値の評価をする
            error, var = evaluate.exe(path, path, xs, xf, ys, yf)
            # error, corr = evaluate.t_test(path, path, xs, xf, ys, yf)
        except:
            print(i)
            continue
        error = error.reshape(1,-1)[0]
        var = var.reshape(1, -1)[0]
        a.append(error)
        b.append(var)

    # # # # # breakpoint()

    # a = np.array(a).reshape(1,-1)[0]
    # b = np.array(b).reshape(1,-1)[0]
    a = np.array(a)
    b = np.array(b)
    np.save('ttest_pe_only_error', a)
    np.save('ttest_pe_only_corr', b)
    # plt.figure(figsize=(16, 5))
    # plt.scatter(b,a)
    # plt.xlabel('variance')
    # plt.ylabel('relative error')
    # plt.savefig('eval3.png' , format="png")
    # # breakpoint()


