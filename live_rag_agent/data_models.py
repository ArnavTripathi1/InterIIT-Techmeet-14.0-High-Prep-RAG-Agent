from dataclasses import dataclass

@dataclass
class Borrower:
    borrower_id: str
    amount: float
    repayments: float
    credit_score_bureau: float