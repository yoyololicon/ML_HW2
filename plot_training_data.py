import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def find_boarder(t):
    b = []
    for i in range(1, len(t)):
        if np.any(t[i, :] - t[i-1, :]):
            b.append(i)

    return b

def plot(axarra, b, c1, c2, c3, variable):
    bins = np.linspace(b[0], b[1], 10)
    axarra.set_title('Variable %d' % variable)
    axarra.hist(c1, bins=bins, alpha=0.5, label='class 1')
    axarra.hist(c2, bins=bins, alpha=0.5, label='class 2')
    axarra.hist(c3, bins=bins, alpha=0.5, label='class 3')
    axarra.legend(loc='upper right')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: %s <train file>' % sys.argv[0])
        sys.exit(1)

    train = pd.read_csv(sys.argv[1]).values
    [train_t, train_x] = np.split(train, [3], axis=1)
    min_max = np.vstack([np.min(train_x, axis=0), np.max(train_x, axis=0)]).T
    boarder = find_boarder(train_t)
    [class1, class2, class3] = np.split(train_x, boarder)

    for i in range(0, 13, 4):
        fig, ax = plt.subplots(2, 2)
        if i < 13:
            plot(ax[0, 0], min_max[i], class1[:, i], class2[:, i], class3[:, i], i)
        if i+1 < 13:
            plot(ax[0, 1], min_max[i+1], class1[:, i+1], class2[:, i+1], class3[:, i+1], i+1)
        if i+2 < 13:
            plot(ax[1, 0], min_max[i+2], class1[:, i+2], class2[:, i+2], class3[:, i+2], i+2)
        if i+3 < 13:
            plot(ax[1, 1], min_max[i+3], class1[:, i+3], class2[:, i+3], class3[:, i+3], i+3)

        plt.show()