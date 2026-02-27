"""
BM25 (Best Matching 25) implementation for document search and retrieval.

This module provides a BM25-based search functionality that can be used to find
the most relevant documents in a corpus based on a query. The implementation
includes NLTK-based text tokenization.
"""

import logging

import bm25s
import Stemmer

logger = logging.getLogger("app")

class BM25:
    """
    BM25 search implementation with S3 model persistence.

    This class provides BM25-based document search functionality. It uses NLTK for text
    tokenization and provides methods for both document retrieval and index-based search.
    """

    def __init__(self, corpus: list[str]):
        """
        Initialize BM25 search with a corpus of documents.

        Args:
            corpus: List of documents to search through
        """
        logger.info("Initializing BM25")

        self.stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def topk_indices(self, query: str, k: int = 10) -> list[int]:
        """
        Get the indices of the top k most relevant documents for a query.

        Args:
            query: Search query string
            k: Number of top results to return (default: 10)

        Returns:
            List of document indices sorted by relevance (highest first)
        """
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        idx, _ = self.retriever.retrieve(query_tokens, k=k)
        return idx[0]

    def encode_token(self, token: str) -> int:
        return self.token_to_code.get(token, self._oov_id)