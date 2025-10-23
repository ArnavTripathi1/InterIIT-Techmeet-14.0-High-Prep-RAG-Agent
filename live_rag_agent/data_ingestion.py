import pandas as pd
from pathlib import Path
from config import TRANSACTIONS_DIR, BUREAU_DIR

def load_data():
    transactions_path = Path(TRANSACTIONS_DIR) / "transactions.csv"
    bureau_path = Path(BUREAU_DIR) / "bureau.csv"

    if not transactions_path.exists() or not bureau_path.exists():
        raise FileNotFoundError("Missing CSV files in data folders.")

    transactions = pd.read_csv(transactions_path)
    bureau = pd.read_csv(bureau_path)

    borrowers = pd.merge(transactions, bureau, on="borrower_id", how="inner")
    return borrowers