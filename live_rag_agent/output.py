def print_results(final_df):
    print("\n=== CREDIT RISK ACTIONS WITH EXPLANATIONS ===\n")
    for _, row in final_df.iterrows():
        print(f"Borrower: {row['borrower_id']}")
        print(f"  Risk: {row['risk']:.2f}")
        print(f"  Action: {row['action']}")
        print(f"  Explanation: {row['explanation']}\n")