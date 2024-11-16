import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
data = pd.read_csv('dataset.csv')

# Preprocessing
data['Start'] = pd.to_datetime(data['Start'], format='%d/%m/%Y %H:%M')
data['End'] = pd.to_datetime(data['End'], format='%d/%m/%Y %H:%M')
data['Duration'] = (data['End'] - data['Start']).dt.total_seconds() / 3600  # Duration in hours

# Features and target variable
X = data[['Device', 'Site', 'StationName', 'Duration']]
y = data['Total_Energy (kWh)']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Device', 'Site', 'StationName'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree regressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotting Predicted vs Actual Values
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Demand', color='blue', marker='o')
plt.plot(y_pred, label='Predicted Demand', color='orange', marker='x')
plt.title('Actual vs Predicted Energy Demand')
plt.xlabel('Samples')
plt.ylabel('Energy Demand (kWh)')
plt.legend()
plt.grid()
plt.show()

# Optionally, visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True)
plt.title('Decision Tree for Energy Prediction')
plt.show()
