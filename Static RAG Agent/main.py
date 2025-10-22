from embeddings import EmbeddingIndex
from rag import query_rag

def main():
    index = EmbeddingIndex()
    print("RAG Agent ready! Type 'exit' to quit.\n")

    while True:
        question = input("Enter your question: ")
        if question.lower() in ["exit", "quit"]:
            break

        retrieved = index.search(question)
        answer = query_rag(question, retrieved)
        print("\nAnswer:", answer)
        print("="*50, "\n")

if __name__ == "__main__":
    main()