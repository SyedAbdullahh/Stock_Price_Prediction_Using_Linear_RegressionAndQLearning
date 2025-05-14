import pickle
import os

# Files to reset
last_update_file = "last_update_date.pkl"
model_file = "stock_model.pkl"
q_table_file = "q_table.pkl"

# Set a past date to force full model retraining
reset_date = "2025-04-01"

# Save the reset date
with open(last_update_file, 'wb') as f:
    pickle.dump(reset_date, f)
print(f"✅ Reset {last_update_file} to {reset_date}")

#Optional: Delete model and Q-table to start fresh
for file in [model_file, q_table_file]:
    if os.path.exists(file):
        os.remove(file)
        print(f"🗑 Deleted {file}")
    else:
        print(f"ℹ {file} does not exist or was already deleted.")