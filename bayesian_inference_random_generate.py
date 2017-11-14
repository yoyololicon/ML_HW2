import numpy as np
from scipy import io
from scipy.stats import wishart
import sys

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: %s <inputfile> <N> <number of random precision>' % sys.argv[0])
        sys.exit(1)

    data = io.loadmat(sys.argv[1])
    for key in data:
        if key[0:2] != '__':
            data = data[key]
            break
    N = int(sys.argv[2])
    rand_p = int(sys.argv[3])

    total, D = data.shape
    true_cov = np.cov(data.T)
    rand_data = data[np.random.choice(data.shape[0], N, replace=False), :]

    w0 = np.identity(D)
    v0 = 1
    mean = np.array([1, -1])

    post_v = v0+N
    post_w = np.linalg.inv(w0)
    for i in range(N):
        post_w += np.outer(rand_data[i]-mean, rand_data[i]-mean)
    post_w = np.linalg.inv(post_w)

    test = wishart.rvs(df=post_v, scale=post_w, size=rand_p)
    probability = np.empty(rand_p)
    print '----testing----'
    for i in range(rand_p):
        probability[i] = wishart.pdf(test[i], df=post_v, scale=post_w)

    pMAP = np.argmax(probability)
    covMAP = np.linalg.inv(test[pMAP])

    print 'the approximated MAP solution of covariance is'
    print covMAP
    print 'the true covariance is'
    print true_cov
    print 'error of approximated MAP solution is', np.mean(np.abs(covMAP - true_cov))
    print 'the posterior probability is', probability[pMAP]