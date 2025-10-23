import pandas as pd
from config import RISK_THRESHOLDS

def compute_risk_scores(df: pd.DataFrame):
    if "amount_due" in df.columns:
        return df["amount_due"] / df["amount_due"].max()
    else:
        import numpy as np
        return pd.Series(np.random.rand(len(df)), index=df.index)

def next_best_action_fn(risk: float) -> str:
    if risk >= RISK_THRESHOLDS["request_collateral"]:
        return "request_collateral"
    elif risk >= RISK_THRESHOLDS["tighten"]:
        return "tighten"
    elif risk >= RISK_THRESHOLDS["review"]:
        return "review"
    else:
        return "approve"