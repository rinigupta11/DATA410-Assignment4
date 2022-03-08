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

Boosting is a popular method to improve the performance of a regressor by turning weaker learners into stronger learners through a sequential process. Boosting once is useful for reducing variance. Some boosting algorithms perform gradient descent with a cost minimization surface. Multiple boosting involves boosting a regressor more than one time to further improve performance because the final regressor is even stronger from all the weaker learners that it combined. I wrote a function that can perform the boosting k number of times. This algorithm uses the locally weighted linear regression model to boost another regression model. In later cases below, we will use other regressors besides lowess for the boosting. Specifically, we initially subtract the lowess prediction values from the target values. We store a variable that will cumulatively change value called output as we create new predictions using the model_boosting model passed in as a parameter. This iterative process results in repeated boosting of the model_boosting regressor passed in. Kaggle has cited multiple boosting and combining different boosting techniques as one of the ways people scored the highest in competitions. 

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
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=1000,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
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

The results are: 

The Cross-validated Mean Squared Error for RF is : 87.96805891018424

The Cross-validated Mean Squared Error for RF (Boosted) is : 38.44758294327894

The Cross-validated Mean Squared Error for Linear Regression is : 115.94705718666674

The Cross-validated Mean Squared Error for Linear Regression (Boosted) is : 54.96532335974295

The Cross-validated Mean Squared Error for SVM is : 85.05483062682657

The Cross-validated Mean Squared Error for SVM (Boosted) is : 53.00967236567205

The Cross-validated Mean Squared Error for XGB is : 28.06138323921298

Here we can see that boosting dramatically improved the performance of the linear regressor, random forest, and support vector machine regression. Extreme gradient boosting was the most effective method out of all that were tested, with a mean-squared error of 28.06. 

We can also try using the random forest regressor as the booster instead of locally weighted linear regression. The function would now look like this.

```
def repeated_boosting(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  rf = RandomForestRegressor(n_estimators = 500, max_depth = 3)
  rf.fit(X,y)
  Fx = rf.predict(X)
  Fx_new = rf.predict(xnew)
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

Running the same cross validation above, we get these results. 

The Cross-validated Mean Squared Error for RF is : 86.8873930596584

The Cross-validated Mean Squared Error for Linear Regression is : 116.05239179871771

The Cross-validated Mean Squared Error for Linear Regression (Boosted) is : 61.397216241160855

The Cross-validated Mean Squared Error for SVM is : 84.27651979574426

The Cross-validated Mean Squared Error for SVM (Boosted) is : 46.512191786451105

The Cross-validated Mean Squared Error for XGB is : 31.08846140470253

Random forest regressor as a booster yielded slightly better results for certain regressors like SVM but extreme gradient boosting is still the winner.


Here are the results when using xgb as the booster. 

```
def repeated_boosting(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
  model_xgb.fit(X,y)
  Fx = model_xgb.predict(X)
  Fx_new = model_xgb.predict(xnew)
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

The Cross-validated Mean Squared Error for RF is : 86.95616892245656

The Cross-validated Mean Squared Error for RF (Boosted) is : 33.624594883976734

The Cross-validated Mean Squared Error for Linear Regression is : 116.05239179871771

The Cross-validated Mean Squared Error for Linear Regression (Boosted) is : 46.69190224432877

The Cross-validated Mean Squared Error for SVM is : 84.27651979574426

The Cross-validated Mean Squared Error for SVM (Boosted) is : 36.99477059591091

The Cross-validated Mean Squared Error for XGB is : 31.08846140470253

XGBoost was a very effective supplement when repeatedly used on different regressors and lowered the MSE values quite substantially. 

### LightGBM
LightGBM is a gradient boosting (tree-based) framework developed by Microsoft to improve upon accuracy, efficiency, and memory-usage of other boosting algorithms. XGBoost is the current star among boosting algorithms in terms of the accuracy that it produces; however, XGBoost can take more time to compute results. As a result, LightGBM aims to compete with its "lighter", speedier framework. LightGBM splits the decision tree by the leaf with the best fit. The way that LightGBM chooses the leaf is by finding the split that will create the greatest loss decrease. In contrast, other boosting algorithms split the tree based on depth. Splitting by the leaf has proven to be a very effective loss reduction technique that boosts accuracy. Furthermore, LightGBM uses a histogram-like approach and puts continuous features into bins to speed training time. This approach has been demonstrated to dramatically increase time and space complexity. The two specific techniques that are part of the LightGBM algorithm are Exclusive Feature Bundling (a feature reduction technique) and Gradient-Based One Side Sampling (higher gradients contribute more information). We will be particularly comparing the accuracy of LightGBM to XGBoost in this paper.

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

The output of this code looks like this: 

Starting training...
[1]	valid_0's l2: 239.673	valid_0's l1: 12.7332
Training until validation scores don't improve for 5 rounds.
[2]	valid_0's l2: 222.898	valid_0's l1: 12.2743
[3]	valid_0's l2: 207.153	valid_0's l1: 11.8424
[4]	valid_0's l2: 192.684	valid_0's l1: 11.4272
[5]	valid_0's l2: 179.863	valid_0's l1: 11.0341
[6]	valid_0's l2: 166.936	valid_0's l1: 10.6174
[7]	valid_0's l2: 156.796	valid_0's l1: 10.2855
[8]	valid_0's l2: 146.236	valid_0's l1: 9.91438
[9]	valid_0's l2: 136.951	valid_0's l1: 9.58067
[10]	valid_0's l2: 128.279	valid_0's l1: 9.25352
[11]	valid_0's l2: 123.497	valid_0's l1: 9.10913
[12]	valid_0's l2: 119.088	valid_0's l1: 8.96037
[13]	valid_0's l2: 112.321	valid_0's l1: 8.67888
[14]	valid_0's l2: 106.101	valid_0's l1: 8.41888
[15]	valid_0's l2: 100.524	valid_0's l1: 8.17604
[16]	valid_0's l2: 95.299	valid_0's l1: 7.96473
[17]	valid_0's l2: 91.2394	valid_0's l1: 7.79198
[18]	valid_0's l2: 86.7809	valid_0's l1: 7.60055
[19]	valid_0's l2: 82.6498	valid_0's l1: 7.41587
[20]	valid_0's l2: 78.9956	valid_0's l1: 7.24818
[21]	valid_0's l2: 75.9804	valid_0's l1: 7.09105
[22]	valid_0's l2: 72.6802	valid_0's l1: 6.92479
[23]	valid_0's l2: 69.2713	valid_0's l1: 6.76221
[24]	valid_0's l2: 66.1247	valid_0's l1: 6.60474
[25]	valid_0's l2: 64.2019	valid_0's l1: 6.51345
[26]	valid_0's l2: 61.6179	valid_0's l1: 6.38057
[27]	valid_0's l2: 59.2139	valid_0's l1: 6.251
[28]	valid_0's l2: 57.7165	valid_0's l1: 6.16519
[29]	valid_0's l2: 55.6803	valid_0's l1: 6.05259
[30]	valid_0's l2: 53.828	valid_0's l1: 5.94143
[31]	valid_0's l2: 51.8483	valid_0's l1: 5.8306
[32]	valid_0's l2: 50.0365	valid_0's l1: 5.71672
[33]	valid_0's l2: 48.7511	valid_0's l1: 5.64612
[34]	valid_0's l2: 47.2473	valid_0's l1: 5.54306
[35]	valid_0's l2: 46.0277	valid_0's l1: 5.46636
[36]	valid_0's l2: 44.9417	valid_0's l1: 5.39845
[37]	valid_0's l2: 43.9986	valid_0's l1: 5.34236
[38]	valid_0's l2: 43.0673	valid_0's l1: 5.2823
[39]	valid_0's l2: 42.0873	valid_0's l1: 5.21709
[40]	valid_0's l2: 41.164	valid_0's l1: 5.16612
[41]	valid_0's l2: 40.4567	valid_0's l1: 5.11271
[42]	valid_0's l2: 39.777	valid_0's l1: 5.06202
[43]	valid_0's l2: 39.1836	valid_0's l1: 5.01585
[44]	valid_0's l2: 38.678	valid_0's l1: 4.97881
[45]	valid_0's l2: 38.0093	valid_0's l1: 4.92521
[46]	valid_0's l2: 37.377	valid_0's l1: 4.87559
[47]	valid_0's l2: 36.5286	valid_0's l1: 4.81787
[48]	valid_0's l2: 36.0772	valid_0's l1: 4.78408
[49]	valid_0's l2: 35.5115	valid_0's l1: 4.74041
[50]	valid_0's l2: 35.0162	valid_0's l1: 4.69805
[51]	valid_0's l2: 34.5237	valid_0's l1: 4.66093
[52]	valid_0's l2: 33.9807	valid_0's l1: 4.61633
[53]	valid_0's l2: 33.4617	valid_0's l1: 4.57293
[54]	valid_0's l2: 33.178	valid_0's l1: 4.54738
[55]	valid_0's l2: 32.7116	valid_0's l1: 4.50741
[56]	valid_0's l2: 32.2789	valid_0's l1: 4.46926
[57]	valid_0's l2: 31.8093	valid_0's l1: 4.43457
[58]	valid_0's l2: 31.4261	valid_0's l1: 4.39836
[59]	valid_0's l2: 31.1961	valid_0's l1: 4.37037
[60]	valid_0's l2: 30.9644	valid_0's l1: 4.34276
[61]	valid_0's l2: 30.6104	valid_0's l1: 4.30998
[62]	valid_0's l2: 30.2449	valid_0's l1: 4.27926
[63]	valid_0's l2: 29.9713	valid_0's l1: 4.25494
[64]	valid_0's l2: 29.7218	valid_0's l1: 4.23257
[65]	valid_0's l2: 29.4651	valid_0's l1: 4.20538
[66]	valid_0's l2: 29.1704	valid_0's l1: 4.17598
[67]	valid_0's l2: 28.865	valid_0's l1: 4.1486
[68]	valid_0's l2: 28.5571	valid_0's l1: 4.12355
[69]	valid_0's l2: 28.3205	valid_0's l1: 4.09828
[70]	valid_0's l2: 27.9693	valid_0's l1: 4.07064
[71]	valid_0's l2: 27.6482	valid_0's l1: 4.03761
[72]	valid_0's l2: 27.4839	valid_0's l1: 4.01844
[73]	valid_0's l2: 27.3262	valid_0's l1: 4.00144
[74]	valid_0's l2: 27.189	valid_0's l1: 3.98459
[75]	valid_0's l2: 26.925	valid_0's l1: 3.96001
[76]	valid_0's l2: 26.7088	valid_0's l1: 3.93318
[77]	valid_0's l2: 26.526	valid_0's l1: 3.91178
[78]	valid_0's l2: 26.2378	valid_0's l1: 3.88545
[79]	valid_0's l2: 26.0913	valid_0's l1: 3.8667
[80]	valid_0's l2: 25.882	valid_0's l1: 3.84436
[81]	valid_0's l2: 25.7031	valid_0's l1: 3.83193
[82]	valid_0's l2: 25.5311	valid_0's l1: 3.82205
[83]	valid_0's l2: 25.3741	valid_0's l1: 3.81207
[84]	valid_0's l2: 25.225	valid_0's l1: 3.80483
[85]	valid_0's l2: 25.0364	valid_0's l1: 3.78731
[86]	valid_0's l2: 24.8443	valid_0's l1: 3.77196
[87]	valid_0's l2: 24.7225	valid_0's l1: 3.75779
[88]	valid_0's l2: 24.5266	valid_0's l1: 3.7391
[89]	valid_0's l2: 24.3697	valid_0's l1: 3.72521
[90]	valid_0's l2: 24.2382	valid_0's l1: 3.71224
[91]	valid_0's l2: 24.0562	valid_0's l1: 3.69408
[92]	valid_0's l2: 23.9617	valid_0's l1: 3.68592
[93]	valid_0's l2: 23.8247	valid_0's l1: 3.67616
[94]	valid_0's l2: 23.6945	valid_0's l1: 3.6651
[95]	valid_0's l2: 23.6542	valid_0's l1: 3.65813
[96]	valid_0's l2: 23.5339	valid_0's l1: 3.64699
[97]	valid_0's l2: 23.4091	valid_0's l1: 3.63623
[98]	valid_0's l2: 23.3403	valid_0's l1: 3.62745
[99]	valid_0's l2: 23.2555	valid_0's l1: 3.62228
[100]	valid_0's l2: 23.1434	valid_0's l1: 3.61021
[101]	valid_0's l2: 23.0055	valid_0's l1: 3.59556
[102]	valid_0's l2: 22.8609	valid_0's l1: 3.58225
[103]	valid_0's l2: 22.7594	valid_0's l1: 3.57439
[104]	valid_0's l2: 22.681	valid_0's l1: 3.56809
[105]	valid_0's l2: 22.6201	valid_0's l1: 3.56167
[106]	valid_0's l2: 22.5319	valid_0's l1: 3.54962
[107]	valid_0's l2: 22.4143	valid_0's l1: 3.53519
[108]	valid_0's l2: 22.3186	valid_0's l1: 3.52114
[109]	valid_0's l2: 22.2394	valid_0's l1: 3.51094
[110]	valid_0's l2: 22.1748	valid_0's l1: 3.50011
[111]	valid_0's l2: 22.1215	valid_0's l1: 3.49406
[112]	valid_0's l2: 22.0106	valid_0's l1: 3.48567
[113]	valid_0's l2: 21.9326	valid_0's l1: 3.47606
[114]	valid_0's l2: 21.8855	valid_0's l1: 3.47127
[115]	valid_0's l2: 21.8031	valid_0's l1: 3.46505
[116]	valid_0's l2: 21.7427	valid_0's l1: 3.45515
[117]	valid_0's l2: 21.6326	valid_0's l1: 3.44288
[118]	valid_0's l2: 21.5175	valid_0's l1: 3.43125
[119]	valid_0's l2: 21.4828	valid_0's l1: 3.42525
[120]	valid_0's l2: 21.4175	valid_0's l1: 3.41731
[121]	valid_0's l2: 21.3879	valid_0's l1: 3.4081
[122]	valid_0's l2: 21.3514	valid_0's l1: 3.40047
[123]	valid_0's l2: 21.3142	valid_0's l1: 3.39348
[124]	valid_0's l2: 21.3223	valid_0's l1: 3.38964
[125]	valid_0's l2: 21.2127	valid_0's l1: 3.3764
[126]	valid_0's l2: 21.1694	valid_0's l1: 3.36697
[127]	valid_0's l2: 21.1387	valid_0's l1: 3.35975
[128]	valid_0's l2: 21.0745	valid_0's l1: 3.35248
[129]	valid_0's l2: 21.0235	valid_0's l1: 3.34633
[130]	valid_0's l2: 20.9519	valid_0's l1: 3.34042
[131]	valid_0's l2: 20.9049	valid_0's l1: 3.33452
[132]	valid_0's l2: 20.7989	valid_0's l1: 3.32364
[133]	valid_0's l2: 20.7466	valid_0's l1: 3.31621
[134]	valid_0's l2: 20.7028	valid_0's l1: 3.3133
[135]	valid_0's l2: 20.6181	valid_0's l1: 3.30492
[136]	valid_0's l2: 20.5601	valid_0's l1: 3.29626
[137]	valid_0's l2: 20.5245	valid_0's l1: 3.29343
[138]	valid_0's l2: 20.5474	valid_0's l1: 3.29288
[139]	valid_0's l2: 20.5209	valid_0's l1: 3.28896
[140]	valid_0's l2: 20.4559	valid_0's l1: 3.28086
[141]	valid_0's l2: 20.3995	valid_0's l1: 3.27783
[142]	valid_0's l2: 20.3446	valid_0's l1: 3.27594
[143]	valid_0's l2: 20.277	valid_0's l1: 3.26988
[144]	valid_0's l2: 20.2262	valid_0's l1: 3.26672
[145]	valid_0's l2: 20.1858	valid_0's l1: 3.26398
[146]	valid_0's l2: 20.2083	valid_0's l1: 3.26664
[147]	valid_0's l2: 20.175	valid_0's l1: 3.2609
[148]	valid_0's l2: 20.135	valid_0's l1: 3.25595
[149]	valid_0's l2: 20.1345	valid_0's l1: 3.25486
[150]	valid_0's l2: 20.1095	valid_0's l1: 3.252
[151]	valid_0's l2: 20.0496	valid_0's l1: 3.24468
[152]	valid_0's l2: 20.0322	valid_0's l1: 3.24192
[153]	valid_0's l2: 20.0058	valid_0's l1: 3.23552
[154]	valid_0's l2: 19.9622	valid_0's l1: 3.23008
[155]	valid_0's l2: 19.9656	valid_0's l1: 3.22988
[156]	valid_0's l2: 19.9938	valid_0's l1: 3.22855
[157]	valid_0's l2: 19.9899	valid_0's l1: 3.22514
[158]	valid_0's l2: 19.9962	valid_0's l1: 3.22231
[159]	valid_0's l2: 19.9998	valid_0's l1: 3.2205
Early stopping, best iteration is:
[154]	valid_0's l2: 19.9622	valid_0's l1: 3.23008
Saving model...
Starting predicting...


The MSE of LightGBM is: 19.962227424914815

LightGBM produced a remarkably low MSE value using the same dataset and standardization procedures, even compared to XGBoost. LightGBM also ran very fast in comparison to the other regressor methods. We conclude that LightGBM is an efficient and effective method to improve regressor scores. 

### References: 
Bachman, E. (2020, March 27). Light GBM vs XGBOOST: Which algorithm takes the Crown. Analytics Vidhya. Retrieved March 6, 2022, from https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/ 

Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". Advances in Neural Information Processing Systems.

Microsoft. (2021, December 26). Lightgbm/simple_example.py at master · Microsoft/Lightgbm. GitHub. Retrieved March 6, 2022, from https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py 

Rocca, J. (2021, March 21). Ensemble methods: Bagging, boosting and stacking. Medium. Retrieved March 6, 2022, from https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205 

Singh, A. (2020, April 20). Boosting algorithms in machine learning. Analytics Vidhya. Retrieved March 6, 2022, from https://www.analyticsvidhya.com/blog/2020/02/4-boosting-algorithms-machine-learning/ 

Welcome to LIGHTGBM's documentation!. Welcome to LightGBM's documentation! - LightGBM 3.3.2.99 documentation. (2022). Retrieved March 6, 2022, from https://lightgbm.readthedocs.io/en/latest/ 



