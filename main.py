import pickle
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix
from datetime import datetime as dt

file_path = "stock_data.csv"

# Function to create and save the initial model using Gradient Descent
def create_model():
    today = "2025-04-01"  # Hardcoding today's date for initial training
    start_date = "2005-01-01"  # Start date for fetching historical data

    # Fetch stock data (AAPL) from Yahoo Finance
    stock_data = yf.download("AAPL", start=start_date, end=today)
    
    print(stock_data)

    stock_data = stock_data.dropna()
    

    # Prepare the data (OHLV)
    X = stock_data[['Open', 'High', 'Low', 'Volume']].values
    y = stock_data['Close'].values.flatten()

    # Initialize the model (Linear Regression model using Gradient Descent)
    model = Model()

    # Train the model using Gradient Descent
    model.train(X, y)

    print("trained")
    
    # Save the model with pickle
    model.save()
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    save_last_update_date(today)

    print("Model created and saved successfully.")




# Class for the Stock Prediction Model (using Linear Regression with Gradient Descent)
class Model:
    def __init__(self):
        self.weights = None
        self.bias = 0
        self.last_updated = None

    def normalize(self, X):
        return (X - self.mean) / self.std

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        # Compute mean and std for normalization
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = self.normalize(X)

        self.weights = np.zeros(X.shape[1])

        for epoch in range(epochs):
            predictions = self.predict(X, normalized=True)
            error = predictions - y
            gradient_w = (2 / X.shape[0]) * np.dot(X.T, error)
            gradient_b = (2 / X.shape[0]) * np.sum(error)

            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b

    def predict(self, X, normalized=False):
        if not normalized:
            X = (X - self.mean) / self.std
        return np.dot(X, self.weights) + self.bias

# The rest of your methods (update, save, load) remain unchanged.

    def update(self, new_data, reward, action):
        """ Update the model using Q-learning reward and action. """
        
        
        
        X_new = new_data[['Open', 'High', 'Low', 'Volume']].values
        y_new = new_data['Close'].values

        # Update model using the new data first
        self.train(X_new, y_new, epochs=100)  # Train the model on the new data

        # After each action (buy, sell, hold), we adjust the model's weights with the reward
        if action == 'buy':
            self.weights += reward * np.mean(X_new, axis=0)  # Adjust weights based on reward
        elif action == 'sell':
            self.weights -= reward * np.mean(X_new, axis=0)  # Adjust weights based on reward

        print(f"Model coefficients adjusted based on the {action} action and reward.")

    def save(self, filename="stock_model.pkl"):
        """ Save the model using Pickle. """
        print("saving...")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print("Model saved.")

    @staticmethod
    def load(filename="stock_model.pkl"):
        """ Load the model from a Pickle file. """
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            # If no model file exists, return a new model
            return Model()


# Save and load the last update date using Pickle
def save_last_update_date(date, filename="last_update_date.pkl"):
    """ Save the last update date using Pickle. """
    with open(filename, 'wb') as f:
        pickle.dump(date, f)
    print(f"Last update date saved as {date}.")

def load_last_update_date(filename="last_update_date.pkl"):
    """ Load the last update date from a Pickle file. """
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None  # If no last update date, return None

def split_data(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Test MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return predictions

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Close', color='blue')
    plt.plot(predictions, label='Predicted Close', color='orange')
    plt.title("Actual vs. Predicted Closing Prices")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Stock Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_confusion_matrix_like(y_true, y_pred, bins=10):
    y_true_binned = pd.cut(y_true, bins=bins, labels=False)
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=False)
    cm = confusion_matrix(y_true_binned, y_pred_binned)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Binned Confusion Matrix (Regression)")
    plt.colorbar()
    plt.xlabel("Predicted Bin")
    plt.ylabel("Actual Bin")
    plt.tight_layout()
    plt.show()
def train_and_test_on_recent_data():
    # Load 2 years of data
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=730)

    df = yf.download("AAPL", start=start_date, end=end_date).dropna()

    X = df[['Open', 'High', 'Low', 'Volume']].values
    y = df['Close'].values.flatten()

    # Split into training and testing
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train using your existing model
    model = Model()
    model.train(X_train, y_train)

    # Evaluate
    predictions = evaluate_model(model, X_test, y_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Test Results ---")
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    accuracy = 100 - mape

    print(f"Accuracy (100 - MAPE): {accuracy:.2f}%")

    # Plot results
    plot_predictions(y_test, predictions)
    plot_confusion_matrix_like(y_test, predictions)


    model.save()

    last_date = dt.now().strftime("%Y-%m-%d")
    print(f"Last update date: {last_date}")
    
    save_last_update_date(last_date)





# Run the update process
def main():
    # Check if model already exists, if not, create it
    
    create_model()
      #model=Model.load()
    # #  #update_model()
    # model=Model.load()
    # features=np.array([[217.06,222.79,216.46,94130000]])
    # predictedModel=model.predict(features)
    # print(predictedModel[0])

    
     
     #print(predictedModel.shape)
    
    # Run the model update process

    # last_date = dt.now().strftime("%Y-%m-%d")
    # print(f"Last update date: {last_date}")

    # train_and_test_on_recent_data()



# Execute the main function
if __name__ == "__main__":
    main()
