import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from documents import chunk_text, load_documents

class EmbeddingIndex:
    def __init__(self, folder="data", similarity_threshold=0.6):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.folder = folder
        self.similarity_threshold = similarity_threshold
        self.all_passages = []
        self.meta = []
        self.index = None
        self.file_set = set()
        self.embeddings_file = "embeddings.npy"
        self.meta_file = "meta.npy"

        self._load_existing_embeddings()
        self._check_new_files_and_update()

    def _load_existing_embeddings(self):
        if os.path.exists(self.embeddings_file) and os.path.exists(self.meta_file):
            self.meta = list(np.load(self.meta_file, allow_pickle=True))
            embeddings = np.load(self.embeddings_file)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings.astype("float32"))
            self.all_embeddings = embeddings
            self.file_set = set([m['doc_id'] for m in self.meta])
            print(f"Loaded {len(self.meta)} passages from disk.")
        else:
            self.index = None
            self.all_embeddings = np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype="float32")

    def _check_new_files_and_update(self):
        docs = load_documents(self.folder)
        new_docs = [doc for doc in docs if doc["id"] not in self.file_set]

        if not new_docs:
            return

        new_passages = []
        new_meta = []
        for doc in new_docs:
            for chunk in chunk_text(doc["text"]):
                new_passages.append(chunk)
                new_meta.append({"doc_id": doc["id"], "text": chunk})
            self.file_set.add(doc["id"])

        if not new_passages:
            return

        new_embeddings = self.model.encode(new_passages, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

        if self.index is None:
            dim = new_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(new_embeddings.astype("float32"))
            self.all_passages = new_passages
            self.meta = new_meta
            self.all_embeddings = new_embeddings
        else:
            self.index.add(new_embeddings.astype("float32"))
            self.all_passages.extend(new_passages)
            self.meta.extend(new_meta)
            self.all_embeddings = np.vstack([self.all_embeddings, new_embeddings])

        np.save(self.embeddings_file, self.all_embeddings)
        np.save(self.meta_file, np.array(self.meta, dtype=object))
        print(f"Embedded {len(new_passages)} new passages. Total passages: {len(self.meta)}")

    def search(self, query, top_k=5):
        
        self._check_new_files_and_update()

        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        D, I = self.index.search(q_emb.astype("float32"), top_k)

        retrieved = []
        for score, idx in zip(D[0], I[0]):
            if score >= self.similarity_threshold:
                retrieved.append(self.meta[idx])

        return retrieved