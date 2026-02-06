import polars as pl
import numpy as np
import logging
import time

from ..types import EmbeddingClient
from .vector_search import VectorSearch
from .bm25 import BM25


logger = logging.getLogger(__name__)


class SearchClient:
    def __init__(self, chunk_df: pl.DataFrame, embeddings: np.ndarray, embedder: EmbeddingClient):
        self.corpus_series = chunk_df["contextual_chunk"]
        self.corpus_size = len(self.corpus_series)
        self.vector_db = VectorSearch(embeddings)
        # Initialize BM25 with lazy corpus loading
        self.bm25 = BM25(self.corpus_series.to_list())

    def search(self, query: str, k: int = 10):
        """
        Perform rank fusion by combining BM25 and vector search results using multithreaded computation.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of top k results ranked by fusion score
        """
        logger = logging.getLogger("app")
        logger.info(
            f"Ranking fusion search for query: {query} (method called at {time.time()})"
        )

        def get_bm25_ranks():
            """Get BM25 rankings in a separate thread."""
            indices = self.bm25.topk_indices(query, 120)
            return {idx: r for r, idx in enumerate(indices)}

        def get_vector_ranks():
            """Get vector search rankings in a separate thread."""
            indices = self.vector_db.topk_indices(query, 120)
            return {idx: r for r, idx in enumerate(indices)}

        logger = logging.getLogger("app")
        t0 = time.perf_counter()
        a_ranks = get_bm25_ranks()
        logger.debug(f"BM25: {time.perf_counter()-t0}")
        t0 = time.perf_counter()
        b_ranks = get_vector_ranks()
        logger.debug(f"Vector: {time.perf_counter()-t0}")

        t0 = time.perf_counter()
        # Combine all unique indices from both rankings
        idxs = list(set([i for i in a_ranks.keys()] + [i for i in b_ranks.keys()]))

        # Apply reciprocal rank fusion formula: 1/(60 + rank_a) + 1/(60 + rank_b)
        # Use get() with default value 121 for indices not in either ranking
        def fusion_score(idx):
            rank_a = a_ranks.get(idx, 121)
            rank_b = b_ranks.get(idx, 121)
            return (1.0 / (60.0 + rank_a)) + (1.0 / (60.0 + rank_b))

        # Sort by fusion score and return top k
        sorted_ranks = sorted(idxs, key=fusion_score, reverse=True)[:k]

        # Convert indices back to records
        records = {
            record["idx"]: record
            for record in self.df.filter(pl.col("idx").is_in(sorted_ranks)).to_dicts()
        }
        logger.debug(f"Rank fusion: {time.perf_counter()-t0}")

        return [records[idx] for idx in sorted_ranks]
