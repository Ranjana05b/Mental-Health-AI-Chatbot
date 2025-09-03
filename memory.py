import faiss
import numpy as np
import os
import pickle

class FAISSMemory:
    def __init__(self, user_id: str, dim: int = 1024):
        """
        Create or load a FAISS memory index for a specific user.
        Each user has their own memory file.
        """
        self.user_id = user_id
        self.dim = dim
        self.index_file = f"memory_{user_id}.index"
        self.store_file = f"memory_{user_id}.pkl"

        if os.path.exists(self.index_file) and os.path.exists(self.store_file):
            print(f"Loading existing memory for {user_id}")
            self.index = faiss.read_index(self.index_file)
            with open(self.store_file, "rb") as f:
                self.texts = pickle.load(f)
        else:
            print(f"Creating new memory for {user_id}")
            self.index = faiss.IndexFlatL2(self.dim)
            self.texts = []

    def add_memory(self, embedding: np.ndarray, role: str, text: str):
        """
        Save a new memory (embedding + role + text).
        """
        embedding = np.array([embedding], dtype="float32")
        self.index.add(embedding)
        self.texts.append((role, text))

        # persist both index & text
        faiss.write_index(self.index, self.index_file)
        with open(self.store_file, "wb") as f:
            pickle.dump(self.texts, f)

        print(f"Saved memory for {self.user_id}: [{role}] {text}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Retrieve top_k closest memories for given embedding.
        """
        if self.index.ntotal == 0:
            return []
        query_embedding = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.texts):
                results.append(self.texts[idx])
        return results
