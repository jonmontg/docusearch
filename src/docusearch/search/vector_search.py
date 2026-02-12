import logging
import time

import faiss
import numpy as np
from numpy.typing import NDArray
from ..types import EmbeddingClient

logger = logging.getLogger(__name__)

class VectorSearch:
    def __init__(self, embeddings: np.ndarray, embedder: EmbeddingClient, normalize: bool = True):
        self.embedder = embedder
        self.embeddings = embeddings
        self.normalize = normalize
        if self.normalize:
            faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(self.embeddings)

    def topk_indices(self, query: str, k: int = 10) -> list[int]:
        t0 = time.perf_counter()
        x = self.embedder.embed(query)
        logger.info(f"Time to embed query: {time.perf_counter()-t0}")
        t0 = time.perf_counter()
        if self.normalize:
            faiss.normalize_L2(x)
        _, indices = self.index.search(np.array([x]), k)
        logger.info(f"Time to search: {time.perf_counter()-t0}")
        return indices[0].tolist()