# Homework 2 for Machine Learning

## Environment

* ubuntu 16.04 LTS
* python2.7.12 (using Pycharm 2017.2.4)
* extra modules: numpy, scipy, pandas

## Usage of each file

### Bayesian Inference for Gaussian (HW 1-2)

For the task 1-2, type the following command:

```
python bayesian_inference.py 1_data.mat
```
It will find the MAP solution of the covariance matrix, and compare it to the true covariance (using numpy.cov).
The output is like the following:
```
the true covariance is
[[ 0.30082961  0.39309777]
 [ 0.39309777  0.89266987]]
< N = 10 >
the MAP solution of covariance is
[[ 0.72927715  1.06310276]
 [ 1.06310276  2.47424115]]
error of MAP solution is 0.837507199995
< N = 100 >
the MAP solution of covariance is
[[ 0.36316943  0.45461111]
 [ 0.45461111  1.03532019]]
error of MAP solution is 0.0820042028936
< N = 500 >
the MAP solution of covariance is
[[ 0.31936913  0.39865164]
 [ 0.39865164  0.88775764]]
error of MAP solution is 0.00863987484804
```
I also try another approach, which will randomly generate muliple covariance matrix, and compute their probability to pick up the maximum one.

```
python bayesian_inference_random_generate.py 1_data.mat <number of data> <number of random covariance matrix>

#the output format
----testing----
the approximated MAP solution of covariance is
[[ 0.31907513  0.39842093]
 [ 0.39842093  0.88619125]]
the true covariance is
[[ 0.30082961  0.39309777]
 [ 0.39309777  0.89266987]]
error of approximated MAP solution is 0.00884261234455
the posterior probability is 15.4547626093
```

### Bayesian Linear Regression (HW 2-2, 2-3)

For the task 2-2, type the following command:
```
python bayesian_linear_regression.py 2_data.mat
```
This will plot four image similar to Fig. 3.9 on the testbook for different N, like the following image:
![](images/blr_1.png)

For the task 2-3, type the following command:
```
python bayesian_lin_regress_show_region.py 2_data.mat
```
This will plot four image similar to Fig. 3.8 on the testbook for different N, like the following image:
![](images/blr_2.png)

### Logistic Regression (HW 3-1~3, 3-5, 3-6)

For the task 3-1 and 3-2, type the following command:
```
python lofistic_regression.py train.csv test.csv
```
This will first plot the cross entropy and accuracy versus number of epochs like the following image:
![](images/logistic_1.png)

And the classification result :
```
The test data classification result is :
[2 2 2 1 2 2 2 2 2 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
```
The number means the target class name by columns.

For the task 3-3, please type the following command:
```
python plot_training_data.py train.csv
```

This will create series of image ploting the distribution of each variable named with the column index:
![](images/hist_1.png)

For the task 3-5 and 3-6, type the following command:
```
python most_contributive_variable.py train.csv test.csv
```

It will first display the following message and plot the distribution of the two variable:
```
The most contributive variable pair is column 3 and 12
```
![](images/3&12.png)

And then redo the task 3-1 & 3-2 again using these two variable.
The test data classification result is:
```
[2 2 2 0 2 2 2 2 2 2 0 1 0 1 1 0 1 1 2 0 1 1 1 0 0 0 1 1 1]
```