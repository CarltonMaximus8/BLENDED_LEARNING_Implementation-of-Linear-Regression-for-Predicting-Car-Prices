# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df =pd.read_csv('CarPrice_Assignment.csv')

df.head()
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df = pd.read_csv('CarPrice_Assignment.csv')
df.head()
X=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
print("Name: Carlton Maximus A")
print("reg no: 212225040052")
print("MODEL COEFFICIENT:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")
print("\nMODEL PERFORMANCE:")
print(F"{'MAE':>12}: {mean_absolute_error(y_test,y_pred):>10.2f}")
print(f"{'RMAE':>12}: {np.sqrt(mean_absolute_error(y_test,y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10.2f}")
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Acutal Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}")
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color' : 'red'}) 
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("q-q Plot")
plt.tight_layout()
plt.show()
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Carlton Maximus A
RegisterNumber:  212225040052
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="1251" height="284" alt="image" src="https://github.com/user-attachments/assets/a844ae53-52c9-4e6d-9b2b-953e8344f01b" />
<img width="848" height="366" alt="image" src="https://github.com/user-attachments/assets/4c4cb4b9-422c-4a9c-9ab1-8424de65c248" />
<img width="1266" height="598" alt="image" src="https://github.com/user-attachments/assets/b795315e-f0f6-475a-96fc-13801d3b1f06" />
<img width="1386" height="551" alt="image" src="https://github.com/user-attachments/assets/f2e42fa3-bdb9-4523-92cc-9273aa8af0f3" />
<img width="880" height="468" alt="image" src="https://github.com/user-attachments/assets/b1dc976d-2818-4595-a686-e54da9345ae0" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
