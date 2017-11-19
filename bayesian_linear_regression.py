import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import spline
import sys

plot_pos = [(10, (0, 0)), (15, (0, 1)), (30, (1, 0)), (80, (1, 1))]

def pos_mapping(number):
    for i in plot_pos:
        if i[0] == number:
            return i[1]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_posterior(a, b, XX, t):
    D = len(XX[0, :])
    SN = np.linalg.inv(a*np.identity(D) + b*np.dot(XX.T, XX))
    mN = b*np.linalg.multi_dot([SN, XX.T, t])
    return mN, SN

def get_basis_form(u, X, s):
    m = len(X)
    n = len(u)
    transform = np.empty([m, n])
    for i in range(m):
        transform[i] = np.array([sigmoid((X[i]-u[j])/s) for j in range(n)])
    return transform

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

    f, axarr = plt.subplots(2, 2)

    for N in test_N:
        pos = pos_mapping(N)
        post_mean, post_var = get_posterior(alpha, beta, get_basis_form(u, X[:N], s), T[:N])

        #generate random samples
        for l in range(5):
            for i in range(30):
                trans_x = np.squeeze(get_basis_form(u, np.array([x[i]]), s))
                rand_mean = np.dot(post_mean.T, trans_x)
                rand_var = 1./beta + np.linalg.multi_dot([trans_x.T, post_var, trans_x])
                y[i] = np.random.normal(rand_mean, rand_var, 1)

            y_smooth = spline(x, y, x_smooth)
            axarr[pos].plot(x_smooth, y_smooth, 'r-')

        axarr[pos].scatter(X[:N], T[:N], s=80, facecolors='none', edgecolors='b')
        axarr[pos].set_title('N = %d' % N)
        axarr[pos].set_xlim(-0.1, 2.1)
        axarr[pos].set_ylim(-15, 15)
        axarr[pos].set_xlabel('x')
        axarr[pos].set_ylabel('t')

    plt.show()