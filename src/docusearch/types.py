from typing import Protocol

from numpy import ndarray


class EmbeddingClient(Protocol):
    """
    Protocol for embedding client classes.

    Any implementation should be a class with an `embed` method that accepts
    a single string and returns a numpy ndarray representing the embedding.
    """

    dimensions: int
    query_rate_limit: int  # requests per minute (when window_seconds=60)
    token_rate_limit: int  # tokens per minute (when window_seconds=60)
    model_id: str

    def embed(self, text: str) -> ndarray:  # pragma: no cover - structural type only
        ...


class QueryModel(Protocol):
    """
    Protocol for a query model.

    Implementations are classes that expose:
    - a `query` method
    - `query_rate_limit` and `token_rate_limit` integer attributes
    """

    query_rate_limit: int # queries per minute
    token_rate_limit: int # tokens per minute
    model_id: str

    def query(self, text: str) -> str:
        ...
