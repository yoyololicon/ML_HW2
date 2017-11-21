import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
import logistic_regression as lr
import plot_training_data as ptd
from itertools import combinations

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: %s <train file> <test file>' % sys.argv[0])
        sys.exit(1)

    train = pd.read_csv(sys.argv[1]).values
    test = pd.read_csv(sys.argv[2]).values
    [train_t, train_x] = np.split(train, [3], axis=1)

    v_set = list(combinations(range(13), 2))
    max_e = 0

    #find variable pair
    for var in v_set:
        W = np.zeros([3, 11])
        new_x = np.delete(train_x, var, axis=1)
        a = np.dot(new_x, W.T)
        y = lr.softmax2(a)
        E = lr.dev1_cross_entropy(y, train_t, new_x)
        H_inv = np.linalg.inv(lr.dev2_cross_entropy(y, new_x))
        W -= np.tensordot(H_inv, E, axes=([1, 3], [0, 1]))
        #finish update w
        a = np.dot(new_x, W.T)
        y = lr.softmax2(a)
        etp = lr.cross_entropy(y, train_t)
        #print 'E(W)', etp
        if etp > max_e:
            max_e = etp
            final_var = var

    print 'The most contributive variable pair is column', final_var[0], 'and', final_var[1]

    boarder = ptd.find_boarder(train_t)

    plt.scatter(train_x[:boarder[0], final_var[0]], train_x[:boarder[0], final_var[1]], s=60, color='b', label='class 1')
    plt.scatter(train_x[boarder[0]:boarder[1], final_var[0]], train_x[boarder[0]:boarder[1], final_var[1]], s=60, color='g', label='class 2')
    plt.scatter(train_x[boarder[1]:-1, final_var[0]], train_x[boarder[1]:-1, final_var[1]], s=60, color='r', label='class 3')
    plt.xlabel('Variable %d' % final_var[0])
    plt.ylabel('Variable %d' % final_var[1])
    plt.legend()
    plt.show()

    stop_c = 83
    entropy = []
    accuracy = []
    parameters = []
    W = np.zeros([3, 2])
    new_x = train_x[:, final_var]

    #redo homework task (1), (2)
    while 1:
        parameters.append(W)
        a = np.dot(new_x, W.T)
        y = lr.softmax2(a)
        entropy.append(lr.cross_entropy(y, train_t))
        accuracy.append(lr.evaluate(y, train_t))
        #print 'E(W)', entropy[-1], 'accuracy', accuracy[-1]
        if entropy[-1] < stop_c:
            break
        if math.isnan(entropy[-1]):
            entropy.pop()
            accuracy.pop()
            parameters.pop()
            break
        E = lr.dev1_cross_entropy(y, train_t, new_x)
        H_inv = np.linalg.inv(lr.dev2_cross_entropy(y, new_x))
        W -= np.tensordot(H_inv, E, axes=([1, 3], [0, 1]))*0.01

    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].plot(accuracy)
    ax[0].set_title('Accuracy')
    ax[1].plot(entropy)
    ax[1].set_title('Cross Entropy')
    ax[1].set_xlabel('Epoch Number')
    ax[1].set_ylabel('Loss')
    plt.show()

    y = lr.softmax2(np.dot(test[:, final_var], parameters[-1].T))
    print 'The test data classification result is :'
    print np.argmax(y, axis=1)