import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import spline
import sys
import bayesian_linear_regression as blr

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: %s <inputfile>' % sys.argv[0])
        sys.exit(1)

    data = loadmat(sys.argv[1])
    X = data['x']
    T = data['t']
    total = len(X)

    test_N = [10, 15, 30, 80]
    M = 7
    s = 0.1
    u = [float(j*2)/M for j in range(M)]
    alpha = 1./math.pow(10, 6)
    beta = 1

    x = np.linspace(0, 2, 30)
    x_smooth = np.linspace(0, 2, 300)
    y = np.empty(30)
    y1 = np.empty(30)
    y2 = np.empty(30)

    f, axarr = plt.subplots(2, 2)

    for N in test_N:
        pos = blr.pos_mapping(N)
        post_mean, post_var = blr.get_posterior(alpha, beta, blr.get_basis_form(u, X[:N], s), T[:N])


        for i in range(30):
            trans_x = np.squeeze(blr.get_basis_form(u, np.array([x[i]]), s))
            y[i] = np.dot(post_mean.T, trans_x)
            rand_var = 1./beta + np.linalg.multi_dot([trans_x.T, post_var, trans_x])
            std = math.sqrt(rand_var)
            y1[i] = y[i] + std
            y2[i] = y[i] - std

        y_smooth = spline(x, y, x_smooth)
        y1_smooth = spline(x, y1, x_smooth)
        y2_smooth = spline(x, y2, x_smooth)
        axarr[pos].plot(x_smooth, y_smooth, 'r-')
        axarr[pos].fill_between(x_smooth, y1_smooth, y2_smooth, facecolor='pink', edgecolor='none')

        axarr[pos].scatter(X[:N], T[:N], s=80, facecolors='none', edgecolors='b')
        axarr[pos].set_title('N = %d' % N)
        axarr[pos].set_xlim(-0.1, 2.1)
        axarr[pos].set_ylim(-15, 15)
        axarr[pos].set_xlabel('x')
        axarr[pos].set_ylabel('t')

    plt.show()