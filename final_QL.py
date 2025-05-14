import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os

ACTIONS = ['buy', 'sell', 'hold']
GAMMA = 0.9       # Discount factor
ALPHA = 0.1       # Learning rate
EPSILON = 0.2     # Exploration rate
Q_TABLE_FILE = 'q_table.pkl'


class Model:
    def __init__(self):
        self.weights = None
        self.bias = 0
        self.mean = None
        self.std = None
        self.last_updated = None
        self.is_trained = False  # New flag to track training status

    def normalize(self, X):
        if self.mean is None or self.std is None:
            return X
        std_safe = np.where(self.std < 1e-6, 1.0, self.std)
        return (X - self.mean) / std_safe


    def train(self, X, y, epochs=1000, learning_rate=0.01):
        # Calculate normalization stats if not exists
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        
        X_norm = self.normalize(X)
        
        # Initialize weights if needed
        if self.weights is None:
            self.weights = np.random.normal(scale=0.01, size=X.shape[1])  # Small random init
        # Clip weights to prevent explosion
        if self.weights is not None:
            self.weights = np.clip(self.weights, -1000, 1000)

            
        for _ in range(epochs):
            predictions = np.dot(X_norm, self.weights) + self.bias
            error = predictions - y.flatten()
            
            # Gradient descent with L2 regularization
            gradient_w = (2/X.shape[0]) * np.dot(X_norm.T, error) + 0.001 * self.weights
            gradient_b = (2/X.shape[0]) * np.sum(error)
            
            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b
        
        self.is_trained = True

    def predict(self, X, normalized=False):
        # if not self.is_trained:
        #     print("Warning: Making prediction with untrained model!")
        #     return np.zeros(X.shape[0])
        if np.any(np.abs(self.weights) > 1e4):
            print("Warning: Abnormal weight values detected.")

        if not normalized:
            X = self.normalize(X)
        return np.dot(X, self.weights) + self.bias

    def update(self, new_data, reward, action):
        X_new = new_data[['Open', 'High', 'Low', 'Volume']].values
        y_new = new_data['Close'].values
        
        # Clip reward to reasonable range
        reward = np.clip(float(reward), -10, 10) if not isinstance(reward, (int, float)) else reward
        
        # Train with new data
        self.train(X_new, y_new, epochs=50)  # Reduced epochs for online learning
        
        # More conservative weight adjustment
        # try:
        #     adjustment = 0.001 * reward * np.mean(X_new, axis=0)  # Small learning rate
        #     if action == 'buy':
        #         self.weights += adjustment
        #     elif action == 'sell':
        #         self.weights -= adjustment
        # except Exception as e:
        #     print("Adjustment error:", e)

# ... [keep other functions the same until predict_stock_price] ...


    def save(self, filename="stock_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename="stock_model.pkl"):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return Model()

def load_q_table():
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_q_table(q_table):
    with open(Q_TABLE_FILE, 'wb') as f:
        pickle.dump(q_table, f)

def get_state(row):
    # Ensure we get a float value
    if isinstance(row, pd.Series):
        pct_change = row['Close_Change']
    else:
        pct_change = row['Close_Change'].iloc[0]
    
    #pct_change = float(pct_change)
    pct_change=float(pct_change.iloc[0])
    
    if pct_change > 0.005:
        return 'up'
    elif pct_change < -0.005:
        return 'down'
    return 'stable'

def choose_action(state, q_table):
    if isinstance(state, (list, np.ndarray)):
        state = tuple(state)
    elif not isinstance(state, (str, tuple)):
        state = str(state)

    if np.random.rand() < EPSILON or state not in q_table:
        return np.random.choice(ACTIONS)
    
    return max(q_table[state].items(), key=lambda x: x[1])[0]

def simulate_reward(prev_price, current_price, action):
    # Ensure we're working with float values
    prev = float(prev_price.iloc[0]) if hasattr(prev_price, 'iloc') else float(prev_price)
    curr = float(current_price.iloc[0]) if hasattr(current_price, 'iloc') else float(current_price)
    
    if action == 'buy':
        return curr - prev
    elif action == 'sell':
        return prev - curr
    return 0.0

def save_last_update_date(date, filename="last_update_date.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(date, f)

def load_last_update_date(filename="last_update_date.pkl"):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return "2020-01-01"

def update_model():
    model = Model.load()
    q_table = load_q_table()
    last_date = load_last_update_date()
    today = datetime.today().strftime('%Y-%m-%d')

    data = yf.download("AAPL", start=last_date, end=today, auto_adjust=True)
    data['Volume'] = data['Volume'] / 1e6  # Scale volume to millions

    if data.empty or len(data) < 2:
        print("Not enough data to update the model.")
        return

    data = data.dropna().reset_index()
    data['Close_Change'] = data['Close'].pct_change().fillna(0)

    print(f"Fetched {len(data)} new rows from {last_date} to {today}")

    for i in range(1, len(data)):
        prev_row = data.iloc[i - 1]
        curr_row = data.iloc[i]
        prev_state = get_state(prev_row)
        curr_state = get_state(curr_row)

        # Initialize Q-table entries if they don't exist
        if prev_state not in q_table:
            q_table[prev_state] = {a: 0.0 for a in ACTIONS}
        if curr_state not in q_table:
            q_table[curr_state] = {a: 0.0 for a in ACTIONS}

        action = choose_action(prev_state, q_table)
        reward = simulate_reward(prev_row['Close'], curr_row['Close'], action)

        # Q-learning update
        q_old = q_table[prev_state][action]
        q_max_next = max(q_table[curr_state].values())
        q_table[prev_state][action] = q_old + ALPHA * (reward + GAMMA * q_max_next - q_old)

        # Update model weights
        try:
            model.update(pd.DataFrame([curr_row]), reward, action)
        except Exception as e:
            print(f"Error updating model: {e}")

    model.save()
    save_q_table(q_table)
    save_last_update_date(today)
    print("Model, Q-table, and update date saved.")

def predict_stock_price(open_price, high_price, low_price, volume):
    model = Model.load()
    features = np.array([[open_price, high_price, low_price, volume / 1e6]])

    
    # if not model.is_trained:
    #     print("Error: Model needs training first! Uncomment update_model()")
    #     return None
        
    prediction = model.predict(features, normalized=False)  # still False, triggers internal normalization
    print(f"Predicted Close Price: ${prediction[0]:.2f} ")
    if prediction[0] < open_price:
        print("Recommended Action: SELL")
    else:
        print("Recommended Action: BUY/HOLD")
    return prediction[0]


if __name__ == "__main__":
    # MUST run this first to train the model
    #update_model() 
    
    # Example prediction (will be bad until model is trained)
    # predict_stock_price(236.81, 242.20, 235.62, 53610000) # Expected Output (Green Candle) : HOLD
    # print("-----------------------------------")
    # predict_stock_price(247.97,255.00,245.77,147500000) # Expected Output (Green Candle) : HOLD
    # print("-----------------------------------")

    # predict_stock_price(211.51,212.53,201.64,101350000) # Expected Output (Red Candle): SELL

    predict_stock_price(234.5,254.90,213.56,1023000)