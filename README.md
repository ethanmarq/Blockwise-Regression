# Blockwise Regression
## Multinomial Logistic Regression
F(W) = (1/n) sum_i softmax_loss_i(W) + lambda1 * ||W||_1 + lambda2/2 * ||W||_F^2
``` shell
run_mlr_comparsion_all('dataset.mat')
```

## Multiresponse Linear Regression
F(X) =  0.5*||A*X - Y||_F^2 + lambda1*||X||_1 + 0.5*lambda2*||X||_F^2
``` shell
compare_linear_regression('dataset.mat')
```
