# 📈 Stock Price Prediction with Q-Learning and Linear Regression

This project demonstrates two approaches to stock price prediction:

* **Linear Regression using Gradient Descent**
* **Q-Learning**, a Reinforcement Learning technique

---

## 📂 Project Structure

```
Stock_Price_Prediction/
│
├── data/                         # Folder containing stock price datasets
│
├── Binned Confusion Matrix.ipynb# (Optional) Visualization for prediction quality
├── changeLastDate.py            # Script to update the latest date in the dataset
├── last_update_date.pkl         # Stores the last date data was updated
│
├── linear_regression.py         # Linear Regression with Gradient Descent
├── main.py                      # Entrypoint for running prediction models
├── q_table.pkl                  # Trained Q-table for reinforcement learning
├── stock_model.pkl              # Trained Linear Regression model
```

---

## 🛠️ Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare dataset**
   Place historical stock data (CSV or similar) into the `data/` directory.

---

## 🚀 How to Run

### 1. Linear Regression

```bash
python linear_regression.py
```

* Implements Linear Regression from scratch using Gradient Descent
* Saves trained model to `stock_model.pkl`

### 2. Q-Learning

```bash
python main.py
```

* Loads historical stock data
* Trains Q-table to maximize future returns based on actions (Buy, Sell, Hold)
* Saves Q-table to `q_table.pkl`

---

## 📊 Evaluation

* **Linear Regression**:

  * Mean Squared Error (MSE)
  * Visualize predictions vs actual prices

* **Q-Learning**:

  * Cumulative rewards per episode
  * Trade accuracy (optional custom metric)
  * Q-table heatmaps (optional)

---

## 📌 Notes

* `changeLastDate.py` can be used to update the date
* This project is for **educational purposes** and may not reflect actual market behavior.
* No financial advice is implied.
