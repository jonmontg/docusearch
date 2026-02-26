"""
docusearch

Core package for document search utilities.
"""

from .docusearch import Docusearch
from .query import RateLimitManager
from .types import EmbeddingClient, QueryModel

__all__ = ["Docusearch", "EmbeddingClient", "QueryModel", "RateLimitManager", "__version__"]

__version__ = "0.1.0"
