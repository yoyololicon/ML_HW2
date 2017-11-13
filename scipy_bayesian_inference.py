import numpy as np
from scipy import io
from scipy.stats import wishart, multivariate_normal
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
    pN = int(sys.argv[3])

    total, D = data.shape
    true_cov = np.cov(data.T)
    rand_data = data[np.random.choice(data.shape[0], N, replace=False), :]

    w0 = np.identity(D)
    v0 = D
    mean = np.array([1, -1])

    rand_p = np.empty([pN, D, D])
    posteriors = np.empty(pN)
    for i in range(pN):
        rand_p[i] = wishart.rvs(df=v0, scale=w0)
        prior = wishart.pdf(rand_p[i], df=v0, scale=w0)
        for j in range(N):
            prior *= multivariate_normal.pdf(rand_data[j], mean=mean, cov=np.linalg.inv(rand_p[i]))
        posteriors[i] = prior

    pMAP = np.argmax(posteriors)
    print 'the MAP solution of covariance is'
    print np.linalg.inv(rand_p[pMAP])
    print 'the true covariance is'
    print true_cov
    print 'the posterior probability of the MAP solution after reading', N, 'input is', posteriors[pMAP]