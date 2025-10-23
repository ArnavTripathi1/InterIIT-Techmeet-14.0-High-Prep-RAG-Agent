from config import POLICY_DIR, EMBEDDINGS_DIR
from rag_explainer import load_or_create_vectorstore, generate_explanations
from actions import compute_risk_scores, next_best_action_fn
import pandas as pd

def main():
    # print("BASE_DIR:", POLICY_DIR.parent)
    # print(f"POLICY_DIR exists? {POLICY_DIR.exists()} -> {POLICY_DIR}")
    # print("EMBEDDINGS_DIR:", EMBEDDINGS_DIR)

    data_path = POLICY_DIR.parent / "transactions/borrowers.csv"
    df = pd.read_csv(data_path)

    print("Computing risk scores...")
    df["risk"] = compute_risk_scores(df)

    print("Generating next best actions...")
    df["action"] = df["risk"].apply(next_best_action_fn)

    print("Setting up RAG explainer...")
    vectorstore = load_or_create_vectorstore()
    final_df = generate_explanations(df, vectorstore)

    print(final_df.head())

if __name__ == "__main__":
    main()