# Multiple Boosting and LightGBM Analysis on Different Regressors
### By Rini Gupta

This paper will examine multiple boosting and its impact on several common regressors. To analyze the efficacy of repeated boosting, I will use the concrete compressive strength dataset. Furthermore, I will utilize the concept of cross-validation to compare mean-squared error values of multiple boosting on different regressors, extreme gradient boosting, and light GBM while providing theoretical background on these algorithms. 

## Import Libraries/Load Data
```
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.svm import SVR
import lightgbm as lgb
```

## Multiple Boosting

Boosting is a popular method to improve the performance of a regressor. Multiple boosting involves boosting a regressor more than one time to further improve performance. I wrote a function that can perform the boosting k number of times. This algorithm uses the locally weighted linear regression model to boost another regression model. Specifically, we initially subtract the lowess prediction values from the target values. We store a variable that will cumulatively change value called output as we create new predictions using the model_boosting model passed in as a parameter. This iterative process results in repeated boosting of the model_boosting regressor passed in. 

```
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 

```

```
#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```

```
def repeated_boosting(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new 
```

We then initialize the regressors that we want to pass in to the repeated boosting function.
```
model_boosting = LinearRegression()
svm = SVR()
rf = RandomForestRegressor(n_estimators=500,max_depth=3)
```

Now, we run nested k-fold cross validation on the original models, the boosted models, and XGBoost to see the average MSE values. 
```
# we want more nested cross-validations
scale = StandardScaler()

boosted_linear = []
boosted_rf = []
boosted_svm = []
mse_xgb = []
linear = []
rf_mse = []
svm_mse = []


for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
    
    
    yhat_linear_boost = repeated_boosting(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
    yhat_rf_boost = repeated_boosting(xtrain,ytrain,xtest,Tricubic,1,True,rf,2)
    yhat_svm_boost = repeated_boosting(xtrain,ytrain,xtest,Tricubic,1,True,svm,2)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    model_boosting.fit(xtrain, ytrain)
    yhat_linear = model_boosting.predict(xtest)
    rf.fit(xtrain, ytrain)
    yhat_rf = rf.predict(xtest)
    svm.fit(xtrain, ytrain)
    yhat_svm = svm.predict(xtest)

    
    boosted_linear.append(mse(ytest,yhat_linear_boost))
    boosted_rf.append(mse(ytest,yhat_rf_boost))
    boosted_svm.append(mse(ytest,yhat_svm_boost))
    mse_xgb.append(mse(ytest,yhat_xgb))
    linear.append(mse(ytest,yhat_linear))
    rf_mse.append(mse(ytest,yhat_rf))
    svm_mse.append(mse(ytest,yhat_svm))
```

The results are: LOREUM IPSUM 

LightGBM is a gradient boosting (tree-based) framework developed by Microsoft to improve upon accuracy, efficiency, and memory-usage of other boosting algorithms. XGBoost is the current star among boosting algorithms in terms of the accuracy that it produces; however, XGBoost can take more time to compute results. As a result, LightGBM aims to compete with its "lighter", speedier framework. LightGBM splits the decision tree by the leaf with the best fit. In contrast, other boosting algorithms split the tree based on depth. Splitting by the leaf has proven to be a very effective loss reduction technique that boosts accuracy. Furthermore, LightGBM uses a histogram-like approach and puts continuous features into bins to speed training time. We will be particularly comparing the accuracy of LightGBM to XGBoost in this paper.

![image](https://user-images.githubusercontent.com/76021844/156649680-3fba1f2b-7054-455a-aed5-0782d030d045.png)

The code below runs LightGBM on our dataset. 

```
xtrain, xtest, ytrain, ytest = tts(X, y)
xtrain = scale.fit_transform(xtrain)
xtest = scale.transform(xtest)

# create dataset for lightgbm
lgb_train = lgb.Dataset(xtrain, ytrain)
lgb_eval = lgb.Dataset(xtest, ytest, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=5)])

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(xtest, num_iteration=gbm.best_iteration)
# eval
mse_test = mse(ytest, y_pred)
print("\n\nThe MSE of LightGBM is:", mse_test)
# Source: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
```
