import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Create training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Create a sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(units=4, input_dim=2, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1000, verbose=0)

# Test the model
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([[0], [1], [1], [0]])
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
