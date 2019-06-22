# --------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error , r2_score
from matplotlib import pyplot as plt
## Load the data
sales_data = pd.read_csv(path)

## Split the data and preprocess
train = sales_data[sales_data.source == 'train'].copy()
test = sales_data[sales_data.source == 'test'].copy()

train.drop(['source'],axis=1,inplace=True)
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
## Baseline regression model
X1 = train.loc[:,['Item_Weight','Item_MRP','Item_Visibility']]
x_train1,x_val1,y_train1,y_val1 = train_test_split(X1,train.Item_Outlet_Sales,test_size=0.3,random_state = 45)


#initiate and fit the model
alg1 = LinearRegression(normalize=True)
alg1.fit(x_train1,y_train1)

#predict on valiation set
y_pred1 = alg1.predict(x_val1)

#rmse
mse1 = mean_squared_error(y_val1,y_pred1)

print('model1 mse',mse1)
print('model 1 r2 score',r2_score(y_val1,y_pred1))


## Effect on R-square if you increase the number of predictors
X2 = train.drop(['Item_Outlet_Sales','Item_Identifier'],axis=1)
x_train2,x_val2,y_train2,y_val2 = train_test_split(X2,train.Item_Outlet_Sales,test_size = 0.3,random_state = 45)

alg2 = LinearRegression().fit(x_train2,y_train2)
y_pred2 = alg2.predict(x_val2)

mse2 = mean_squared_error(y_val2,y_pred2)

print('mse2 ',mse2)
print('model 2 r2 score ',r2_score(y_val2,y_pred2))


## Effect on R-square if you increase the number of predictors
X3 = train.drop(['Item_Outlet_Sales','Item_Identifier','Item_Visibility','Outlet_Years'],axis=1)
x_train3,x_val3,y_train3,y_val3 = train_test_split(X3,train.Item_Outlet_Sales,test_size = 0.3,random_state = 45)

alg3 = LinearRegression().fit(x_train3,y_train3)
y_pred3 = alg3.predict(x_val3)

mse3 = mean_squared_error(y_val3,y_pred3)

print('mse3 ',mse2)
print('model 3 r2 score ',r2_score(y_val3,y_pred3))

## Detecting hetroskedacity
plt.scatter(y_pred2,y_val2 - y_pred2)
plt.hlines(y=0,xmin=-1000,xmax=5000)
plt.title('residual plot')
plt.xlabel('predicted Values')
plt.ylabel('Residuals')


## Model coefficients
coef = pd.Series(alg2.coef_,x_train2.columns).sort_values()

plt.figure(figsize = (10,10))
coef.plot(kind='bar',title = 'Model Coeff')

## Ridge regression
l1 = Lasso(alpha = 0.01).fit(x_train2,y_train2)
l2 = Ridge(alpha = 0.05).fit(x_train2,y_train2)

l1_pred2 = l1.predict(x_val2)
l2_pred2 = l2.predict(x_val2)

l1_mse2 = mean_squared_error(y_val2,l1_pred2)
l2_mse2 = mean_squared_error(y_val2,l2_pred2)

print('Lasso mse',l1_mse2)
print('Lasso r2',r2_score(y_val2,l1_pred2))

print('Ridge Mse',l2_mse2)
print('Ridge r2',r2_score(y_val2,l2_pred2))


## Cross vallidation
from sklearn import model_selection
import numpy as np
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

y_train = y_train2
X_train = x_train2
y_test = y_val2
X_test = x_val2


alpha_vals_lasso = [0.01,0.05,0.5,5]
alpha_vals_ridge = [0.01,0.05,0.5,5,10,15,25]

ridge_model = Ridge()
lasso_model = Lasso()

ridge_grid = GridSearchCV(estimator=ridge_model,param_grid=dict(alpha=alpha_vals_ridge))
ridge_grid.fit(X_train,y_train)

lasso_grid = GridSearchCV(estimator=lasso_model,param_grid=dict(alpha=alpha_vals_lasso))
lasso_grid.fit(X_train,y_train)

#make predictions

ridge_pred = ridge_grid.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test,ridge_pred))

lasso_pred = lasso_grid.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test,lasso_pred))

best_model,Best_Model = ('LASSO',lasso_grid) if lasso_rmse < ridge_rmse else ("RIDGE",ridge_grid)

if best_model == 'LASSO':
    print(best_model,Best_Model.best_params_,lasso_rmse,r2_score(y_test,lasso_pred))
else:
    print(best_model,Best_Model.best_params_,ridge_rmse,r2_score(y_test,ridge_pred))



