import threading
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import EmbeddingClient


class EmbeddingManager:
    """
    Thread-safe manager for EmbeddingClient that enforces rate limits.
    
    Ensures that query_rate_limit is respected across multiple threads
    making concurrent embedding requests.
    """
    
    def __init__(self, embedding_client: "EmbeddingClient", window_seconds: int = 60):
        """
        Initialize EmbeddingManager with an EmbeddingClient.
        
        Args:
            embedding_client: The EmbeddingClient instance to manage
            window_seconds: Time window in seconds for rate limiting (default: 60)
        """
        self.embedding_client = embedding_client
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        
        # Track embedding request timestamps in sliding window
        self._request_timestamps: deque[float] = deque()
    
    def _clean_old_entries(self, current_time: float) -> None:
        """Remove entries outside the current time window."""
        cutoff_time = current_time - self.window_seconds
        
        # Clean request timestamps
        while self._request_timestamps and self._request_timestamps[0] < cutoff_time:
            self._request_timestamps.popleft()
    
    def _wait_for_slot(self, current_time: float) -> float:
        """
        Calculate wait time if necessary to respect query_rate_limit.
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        if len(self._request_timestamps) >= self.embedding_client.query_rate_limit:
            # Need to wait until oldest request falls out of window
            oldest_request_time = self._request_timestamps[0]
            wait_time = (oldest_request_time + self.window_seconds) - current_time
            return max(0.0, wait_time)
        return 0.0
    
    def embed(self, text: str):
        """
        Execute an embedding request through the EmbeddingClient, respecting rate limits.
        
        This method is thread-safe and will block if necessary to respect
        query_rate_limit.
        
        Args:
            text: The text to embed
            
        Returns:
            The result from embedding_client.embed(text)
        """
        # Retry loop to handle race conditions when multiple threads wait
        while True:
            current_time = time.time()
            
            with self._lock:
                # Clean old entries
                self._clean_old_entries(current_time)
                current_time = time.time()  # Update after cleanup
                
                # Calculate wait time
                wait_time = self._wait_for_slot(current_time)
            
            # Sleep outside the lock to avoid blocking other threads
            if wait_time > 0:
                time.sleep(wait_time)
                # Continue loop to re-check after waiting
                continue
            
            # Try to record the request
            with self._lock:
                # Re-check conditions after acquiring lock (another thread may have taken the slot)
                current_time = time.time()
                self._clean_old_entries(current_time)
                current_time = time.time()
                
                wait_time = self._wait_for_slot(current_time)
                
                if wait_time > 0:
                    # Need to wait more, continue loop
                    continue
                
                # Record this request
                self._request_timestamps.append(current_time)
            
            # Execute embedding request outside the lock to avoid blocking other threads
            break
        
        return self.embedding_client.embed(text)
