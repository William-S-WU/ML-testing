#William Schmitt
#1/21/2025
import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

print("Data preprocessing compleate")

# Create the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the training data
#clf.fit(X_train_scaled, y_train)

#y_pred = clf.predict(X_test_scaled)

########################################################################################################################

# Find the Best hyperperameter with a grid search

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up the Grid Search
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform the Grid Search on the training data
grid_search.fit(X_train_scaled, y_train)

# Get the best estimator
best_clf = grid_search.best_estimator_


y_pred = best_clf.predict(X_test_scaled)

###################################

column_names = X.columns.tolist()
print(column_names, type(column_names))
#column_names.remove('quality')
for i in range(len(column_names)):
    print(column_names[i])
    # Extract feature importances
    importances = best_clf.feature_importances_
    feature_names = X.columns #feature_names = X.drop(column_names[i], axis=1).columns  # Using the original feature names

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances using matplotlib
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.show()
#


###################################

'''
##########
# Extract feature importances
importances = best_clf.feature_importances_
feature_names = wine_data.drop('target', axis=1).columns  # Using the original feature names

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances using matplotlib
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()

##########
'''


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


############################################################################


corr_matrix = wine_data.corr()

# Plot the heatmap using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Matrix Heatmap')
plt.show()

