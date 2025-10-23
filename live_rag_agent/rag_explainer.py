import os
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from config import POLICY_DIR, EMBEDDINGS_DIR, LLM_MODEL_PATH, EMBEDDING_MODEL_NAME

def setup_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_policy_metadata_path():
    return EMBEDDINGS_DIR / "policy_metadata.pkl"

def load_metadata():
    path = get_policy_metadata_path()
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def save_metadata(metadata):
    with open(get_policy_metadata_path(), "wb") as f:
        pickle.dump(metadata, f)

def get_file_mod_time(file_path):
    return os.path.getmtime(file_path)

def load_or_create_vectorstore():
    faiss_path = EMBEDDINGS_DIR / "policy_store.faiss"
    embeddings = setup_embeddings()
    stored_metadata = load_metadata()

    new_docs = []
    current_metadata = {}

    print(f"Looking for PDFs in: {POLICY_DIR.resolve()}")
    pdf_files = list(POLICY_DIR.glob("*.pdf"))
    print(f"Found PDFs: {[str(f) for f in pdf_files]}")

    for file in pdf_files:
        mod_time = get_file_mod_time(file)
        current_metadata[str(file)] = mod_time
        if str(file) not in stored_metadata or stored_metadata[str(file)] < mod_time:
            print(f"Loading PDF: {file}")
            loader = PyMuPDFLoader(str(file))
            new_docs.extend(loader.load())

    if not faiss_path.exists() and not new_docs:
        raise RuntimeError(f"No policy documents found to index in {POLICY_DIR}")

    if not faiss_path.exists():
        print("Creating FAISS vectorstore from PDFs...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(new_docs)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(EMBEDDINGS_DIR)
        save_metadata(current_metadata)
    else:
        vectorstore = FAISS.load_local(EMBEDDINGS_DIR, embeddings, allow_dangerous_deserialization=True)
        if new_docs:
            print(f"Updating FAISS with {len(new_docs)} new/modified PDFs...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(new_docs)
            vectorstore.add_documents(split_docs)
            vectorstore.save_local(EMBEDDINGS_DIR)
            save_metadata(current_metadata)
        else:
            print("FAISS up-to-date. No new documents found.")

    return vectorstore

def explain_action(borrower_id, action, risk, retriever):
    
    if hasattr(retriever, "get_relevant_documents"):
        relevant_docs = retriever.get_relevant_documents(f"Policy for {action}")
    else:
        relevant_docs = retriever._get_relevant_documents(f"Policy for {action}", run_manager=None)

    context = "\n".join([d.page_content for d in relevant_docs])

    prompt = f"""
Borrower {borrower_id} has risk={risk:.2f} and suggested action: {action}.
Explain the reasoning and provide relevant policy guidance.
Policy context:
{context}
"""
    llm = GPT4All(model=str(LLM_MODEL_PATH), backend="gptj", n_threads=4)
    return llm(prompt)

def generate_explanations(actions_df, vectorstore):
    retriever = vectorstore.as_retriever()
    actions_df["explanation"] = actions_df.apply(
        lambda row: explain_action(row["borrower_id"], row["action"], row["risk"], retriever),
        axis=1
    )
    return actions_df