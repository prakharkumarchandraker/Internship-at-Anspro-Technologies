import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
warnings.filterwarnings(action = 'ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

# Load the LPG data
LPG_DF = pd.read_csv("./LPG_Data1.csv")

# Split the data into features (X) and target (y)
X = LPG_DF[['CUSTOMER_ID', 'MEMBERS']].values
y = LPG_DF['LPG_RESULT'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# Train the SVR model
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

# Evaluate the SVR model using cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
svr_scores = cross_val_score(regressor, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
svr_scores = np.absolute(svr_scores)
print('SVR Mean MAE: %.3f (%.3f)' % (np.mean(svr_scores), np.std(svr_scores)))

# Make predictions using the SVR model on the test set
svr_preds = regressor.predict(X_test)

# Train the Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate the Ridge Regression model using cross-validation
ridge_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
ridge_scores = np.absolute(ridge_scores)
print('Ridge Mean MAE: %.3f (%.3f)' % (np.mean(ridge_scores), np.std(ridge_scores)))

# Make predictions using the Ridge Regression model on the test set
ridge_preds = model.predict(X_test)

# Compare the performance of the models on the test set
print('SVR MAE on test set: %.3f' % metrics.mean_absolute_error(y_test, svr_preds))
print('Ridge MAE on test set: %.3f' % metrics.mean_absolute_error(y_test, ridge_preds))

# Visualize the predictions of the two models on the test set
plt.scatter(y_test, svr_preds, color="blue", label="SVR")
plt.scatter(y_test, ridge_preds, color="red", label="Ridge")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.legend()
plt.show()
