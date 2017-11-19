import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

train = pd.read_csv('Data/train.csv').values
[train_t, train_x] = np.split(train, [3], axis=1)
min_max = np.vstack([np.min(train_x, axis=0), np.max(train_x, axis=0)]).T
[class1, class2, class3] = np.split(train_x, [49, 98])

for i in range(13):
	bins = np.linspace(min_max[i, 0], min_max[i, 1], 10)
	plt.hist(class1[:, i], bins=bins)
	plt.hist(class2[:, i], bins=bins)
	plt.hist(class3[:, i], bins=bins)
	plt.show()