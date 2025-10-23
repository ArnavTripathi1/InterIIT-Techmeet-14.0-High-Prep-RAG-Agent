import pandas as pd

def compute_risk_fn(row):
    score = 0
    if row.get("amount", 0) > 50000:
        score += 0.3
    if row.get("delinquency", 0) > 1:
        score += 0.5
    return min(score, 1.0)

def compute_risk_table(borrowers: pd.DataFrame):
    borrowers["risk"] = borrowers.apply(compute_risk_fn, axis=1)
    return borrowers[["borrower_id", "risk"]]