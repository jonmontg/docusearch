import threading
import time
from collections import deque
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import EmbeddingClient, QueryModel


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.

    Uses a simple heuristic: ~4 characters per token on average.
    This is a rough estimate and may vary by model.
    """
    return max(1, len(text) // 4)


class RateLimitManager:
    """
    Thread-safe manager that enforces query_rate_limit and token_rate_limit.

    Works with both QueryModel (use .query()) and EmbeddingClient (use .embed()).
    Ensures limits are respected across multiple threads.
    """

    def __init__(self, client: "QueryModel | EmbeddingClient", window_seconds: int = 60):
        """
        Initialize RateLimitManager with a rate-limited client.

        Args:
            client: A QueryModel (has .query) or EmbeddingClient (has .embed)
            window_seconds: Time window in seconds for rate limiting (default: 60)
        """
        self._client = client
        self.window_seconds = window_seconds
        self._lock = threading.Lock()

        self._query_timestamps: deque[float] = deque()
        self._token_usage: deque[tuple[float, int]] = deque()

    @property
    def client(self) -> "QueryModel | EmbeddingClient":
        """The wrapped client (QueryModel or EmbeddingClient)."""
        return self._client

    def _clean_old_entries(self, current_time: float) -> None:
        cutoff_time = current_time - self.window_seconds
        while self._query_timestamps and self._query_timestamps[0] < cutoff_time:
            self._query_timestamps.popleft()
        while self._token_usage and self._token_usage[0][0] < cutoff_time:
            self._token_usage.popleft()

    def _wait_for_query_slot(self, current_time: float) -> float:
        if len(self._query_timestamps) >= self._client.query_rate_limit:
            oldest = self._query_timestamps[0]
            wait_time = (oldest + self.window_seconds) - current_time
            return max(0.0, wait_time)
        return 0.0

    def _wait_for_token_capacity(self, current_time: float, required_tokens: int) -> float:
        current_tokens = sum(tokens for _, tokens in self._token_usage)
        if current_tokens + required_tokens <= self._client.token_rate_limit:
            return 0.0
        tokens_to_free = (current_tokens + required_tokens) - self._client.token_rate_limit
        freed_tokens = 0
        wait_until = current_time
        for timestamp, token_count in self._token_usage:
            if freed_tokens >= tokens_to_free:
                break
            freed_tokens += token_count
            wait_until = timestamp + self.window_seconds
        return max(0.0, wait_until - current_time)

    def _execute(self, text: str, method_name: str) -> Any:
        """Run rate-limited logic then call client method (e.g. 'query' or 'embed')."""
        required_tokens = _estimate_tokens(text)

        while True:
            current_time = time.time()

            with self._lock:
                self._clean_old_entries(current_time)
                current_time = time.time()
                query_wait = self._wait_for_query_slot(current_time)
                token_wait = self._wait_for_token_capacity(current_time, required_tokens)
                max_wait = max(query_wait, token_wait)

            if max_wait > 0:
                time.sleep(max_wait)
                continue

            with self._lock:
                current_time = time.time()
                self._clean_old_entries(current_time)
                current_time = time.time()
                query_wait = self._wait_for_query_slot(current_time)
                token_wait = self._wait_for_token_capacity(current_time, required_tokens)
                if query_wait > 0 or token_wait > 0:
                    continue
                self._query_timestamps.append(current_time)
                self._token_usage.append((current_time, required_tokens))

            break

        return getattr(self._client, method_name)(text)

    def query(self, text: str) -> str:
        """Execute a query (for QueryModel). Thread-safe, rate-limited."""
        return self._execute(text, "query")

    def embed(self, text: str) -> Any:
        """Execute an embed (for EmbeddingClient). Thread-safe, rate-limited."""
        return self._execute(text, "embed")
