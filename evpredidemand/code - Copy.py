import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import matplotlib.pyplot as plt

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

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data for GRU [samples, time steps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build the GRU model
model = Sequential()
model.add(GRU(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = np.mean((y_test - y_pred.flatten())**2)
print(f'Mean Squared Error: {mse}')

# Plotting training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

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
