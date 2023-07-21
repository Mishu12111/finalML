import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv('features.csv')
X = df.drop(['manele'], axis=1)
y = df['manele']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
model_rf = RandomForestRegressor(max_depth=5, n_estimators=300, random_state=42)
model_rf.fit(X_train, y_train)

# Linear Regression
model_lr = LinearRegression(fit_intercept=False)

model_lr.fit(X_train, y_train)

# Saving the models
with open('csharp_rf.pkl', 'wb') as file_rf:
    pickle.dump(model_rf, file_rf)

with open('csharp_lr.pkl', 'wb') as file_lr:
    pickle.dump(model_lr, file_lr)
