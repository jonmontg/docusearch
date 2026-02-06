import threading
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import QueryModel


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    
    Uses a simple heuristic: ~4 characters per token on average.
    This is a rough estimate and may vary by model.
    """
    return max(1, len(text) // 4)


class QueryManager:
    """
    Thread-safe manager for QueryModel that enforces rate limits.
    
    Ensures that query_rate_limit and token_rate_limit are respected
    across multiple threads making concurrent requests.
    """
    
    def __init__(self, query_model: "QueryModel", window_seconds: int = 60):
        """
        Initialize QueryManager with a QueryModel.
        
        Args:
            query_model: The QueryModel instance to manage
            window_seconds: Time window in seconds for rate limiting (default: 60)
        """
        self.query_model = query_model
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        
        # Track query timestamps in sliding window
        self._query_timestamps: deque[float] = deque()
        
        # Track token usage: list of (timestamp, token_count) tuples
        self._token_usage: deque[tuple[float, int]] = deque()
    
    def _clean_old_entries(self, current_time: float) -> None:
        """Remove entries outside the current time window."""
        cutoff_time = current_time - self.window_seconds
        
        # Clean query timestamps
        while self._query_timestamps and self._query_timestamps[0] < cutoff_time:
            self._query_timestamps.popleft()
        
        # Clean token usage
        while self._token_usage and self._token_usage[0][0] < cutoff_time:
            self._token_usage.popleft()
    
    def _wait_for_query_slot(self, current_time: float) -> float:
        """
        Calculate wait time if necessary to respect query_rate_limit.
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        if len(self._query_timestamps) >= self.query_model.query_rate_limit:
            # Need to wait until oldest query falls out of window
            oldest_query_time = self._query_timestamps[0]
            wait_time = (oldest_query_time + self.window_seconds) - current_time
            return max(0.0, wait_time)
        return 0.0
    
    def _wait_for_token_capacity(self, current_time: float, required_tokens: int) -> float:
        """
        Calculate wait time if necessary to respect token_rate_limit.
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        current_tokens = sum(tokens for _, tokens in self._token_usage)
        
        if current_tokens + required_tokens > self.query_model.token_rate_limit:
            # Need to wait until enough tokens are available
            # Calculate when tokens will be available
            tokens_to_free = (current_tokens + required_tokens) - self.query_model.token_rate_limit
            
            # Find the earliest time when enough tokens will be freed
            freed_tokens = 0
            wait_until = current_time
            for timestamp, token_count in self._token_usage:
                if freed_tokens >= tokens_to_free:
                    break
                freed_tokens += token_count
                wait_until = timestamp + self.window_seconds
            
            wait_time = wait_until - current_time
            return max(0.0, wait_time)
        return 0.0
    
    def query(self, text: str) -> str:
        """
        Execute a query through the QueryModel, respecting rate limits.
        
        This method is thread-safe and will block if necessary to respect
        query_rate_limit and token_rate_limit.
        
        Args:
            text: The query text to process
            
        Returns:
            The result from query_model.query(text)
        """
        required_tokens = _estimate_tokens(text)
        
        # Retry loop to handle race conditions when multiple threads wait
        while True:
            current_time = time.time()
            
            with self._lock:
                # Clean old entries
                self._clean_old_entries(current_time)
                current_time = time.time()  # Update after cleanup
                
                # Calculate wait times
                query_wait = self._wait_for_query_slot(current_time)
                token_wait = self._wait_for_token_capacity(current_time, required_tokens)
                max_wait = max(query_wait, token_wait)
            
            # Sleep outside the lock to avoid blocking other threads
            if max_wait > 0:
                time.sleep(max_wait)
                # Continue loop to re-check after waiting
                continue
            
            # Try to record the query
            with self._lock:
                # Re-check conditions after acquiring lock (another thread may have taken the slot)
                current_time = time.time()
                self._clean_old_entries(current_time)
                current_time = time.time()
                
                query_wait = self._wait_for_query_slot(current_time)
                token_wait = self._wait_for_token_capacity(current_time, required_tokens)
                
                if query_wait > 0 or token_wait > 0:
                    # Need to wait more, continue loop
                    continue
                
                # Record this query
                self._query_timestamps.append(current_time)
                self._token_usage.append((current_time, required_tokens))
            
            # Execute query outside the lock to avoid blocking other threads
            break
        
        return self.query_model.query(text)
