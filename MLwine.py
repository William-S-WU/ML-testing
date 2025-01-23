#William Schmitt
#1/21/2025
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
print("Current working directory:", os.getcwd())
directory = os.getcwd()
filename = os.path.join(directory, "wine-quality.csv")
print(filename)
print(type(filename))
wine_data = pd.read_csv(filename)
print(wine_data.head())

# Separate features and target variable
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data splitting and normalization complete")



########################################################################################################################

# Find the Best hyperperameter with a grid search



# Create the KNN classifier
knn = KNeighborsClassifier()

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# Set up Grid Search
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform the Grid Search on the training data
grid_search_knn.fit(X_train_scaled, y_train)

# Get the best estimator
best_knn = grid_search_knn.best_estimator_

y_pred = best_knn.predict(X_test_scaled)


########################################################################################################################
#metrics testing 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
print(f'Precision: {precision:.2f}')

recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
print(f'Recall: {recall:.2f}')

f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f'F1-Score: {f1:.2f}')


print("code finished running")
