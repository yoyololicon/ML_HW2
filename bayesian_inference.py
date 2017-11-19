import numpy as np
from scipy import io
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: %s <inputfile>' % sys.argv[0])
        sys.exit(1)

    data = io.loadmat(sys.argv[1])
    for key in data:
        if key[0:2] != '__':
            data = data[key]
            break
    test_N = [10, 100, 500]

    total, D = data.shape
    if total < test_N[-1]:
        print('Error: the number of data should bigger than %d' % test_N[-1])
        sys.exit(1)
    true_cov = np.cov(data.T)

    prior_w = np.identity(D)
    prior_v = 1
    mean = np.array([1, -1])

    print 'the true covariance is'
    print true_cov

    for N in test_N:
        print '< N =', N, '>'
        post_v = prior_v+N
        post_w = np.linalg.inv(prior_w) + np.dot((data[:N] - mean).T, data[:N] - mean)
        post_w = np.linalg.inv(post_w)

        covMAP = np.linalg.inv(post_w*(post_v-D-1))
        print 'the MAP solution of covariance is'
        print covMAP
        print 'Error of MAP solution is', np.mean(np.abs(covMAP-true_cov))