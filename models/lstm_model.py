# =========================
# LSTM Stock Price Prediction
# =========================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout    

# Load the dataset
data = pd.read_csv('TCS_stock_history.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use Close price only
data = data[['Close']]

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# Create sequences
def create_sequences(data, steps=60):
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32)

print("Final training loss:", history.history['loss'][-1])


# Predict
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# Evaluation
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print("LSTM RMSE:", rmse)


def forecast_future_prices(model, last_sequence, future_days, scaler):
    """
    Forecast future stock prices using trained LSTM model
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        future_predictions.append(pred[0, 0])

        # slide window
        current_sequence = np.append(
            current_sequence[1:], pred[0, 0]
        )

    # Inverse scaling
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    return future_predictions



# Forecast next 30 days
future_days = 30
last_sequence = X_test[-1]

future_prices = forecast_future_prices(
    model,
    last_sequence,
    future_days,
    scaler
)

plt.figure(figsize=(12, 6))

# Historical prices
plt.plot(
    scaler.inverse_transform(y_test.reshape(-1, 1)),
    label="Actual Prices"
)

# Future forecast
plt.plot(
    range(len(y_test), len(y_test) + future_days),
    future_prices,
    label="Future Forecast",
    linestyle="dashed"
)

plt.title("TCS Stock Price Forecast (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


print("Final training loss:", history.history['loss'][-1])
