import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys

def cross_entropy(y, t):
    return -np.sum(t * np.log(y))

def dev1_cross_entropy(y, t, x):
    return np.dot((y - t).T, x)

def get_R(y, k, j):
    return np.diag(y[:, k] * (np.identity(y.shape[1])[k, j] - y[:, j]))

def dev2_cross_entropy(y, x):
    m = x.shape[1]
    k = y.shape[1]
    rt = np.zeros([k, k, m, m])
    for i in range(k):
        for j in range(k):
            R = get_R(y, i, j)
            rt[j, i] = np.linalg.multi_dot([x.T, R, x])
    return rt

def softmax(a):
    rt = np.exp(a)
    return (rt.T / np.sum(rt, axis=1)).T

def softmax2(a):
    rt = np.empty(a.shape)
    for i in range(len(a)):
        for k in range(len(a[i])):
            rt[i, k] = 1. / np.sum(np.exp(a[i] - a[i, k]))
    return rt

def evaluate(y, t):
    yi = np.argmax(y, axis=1)
    ti = np.argmax(t, axis=1)
    return float(np.count_nonzero((yi - ti) == 0)) / len(ti)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: %s <train file> <test file>' % sys.argv[0])
        sys.exit(1)

    train = pd.read_csv(sys.argv[1]).values
    test = pd.read_csv(sys.argv[2]).values
    [train_t, train_x] = np.split(train, [3], axis=1)

    W = np.zeros([train_t.shape[1], train_x.shape[1]])
    stop_c = 40
    entropy = []
    accuracy = []
    parameters = []

    while 1:
        parameters.append(W)
        a = np.dot(train_x, W.T)
        y = softmax2(a)
        entropy.append(cross_entropy(y, train_t))
        accuracy.append(evaluate(y, train_t))
        #print 'E(W)', entropy[-1], 'accuracy', accuracy[-1]
        if entropy[-1] < stop_c:
            break
        if math.isnan(entropy[-1]):
            entropy.pop()
            accuracy.pop()
            parameters.pop()
            break
        E = dev1_cross_entropy(y, train_t, train_x)
        H_inv = np.linalg.inv(dev2_cross_entropy(y, train_x))
        W -= np.tensordot(H_inv, E, axes=([1, 3], [0, 1]))*0.01

    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].plot(accuracy)
    ax[0].set_title('Accuracy')
    ax[1].plot(entropy)
    ax[1].set_title('Cross Entropy')
    ax[1].set_xlabel('Epoch Number')
    ax[1].set_ylabel('Loss')
    plt.show()

    y = softmax2(np.dot(test, parameters[-1].T))
    print 'The test data classification result is :'
    print np.argmax(y, axis=1)