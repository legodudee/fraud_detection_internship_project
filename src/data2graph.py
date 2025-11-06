# import pandas as pd

# file_path = 'data/synthetic_fraud_data.csv'
# transaction_data = pd.read_csv(file_path)

# transaction_data["transaction_id"] = transaction_data.index
# transaction_data["timestamp"] = pd.to_datetime(transaction_data["timestamp"], format='ISO8601')

from datasets import load_dataset

nigerian_ts = load_dataset("electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset")
nigerian_ts = nigerian_ts['train'].to_pandas()