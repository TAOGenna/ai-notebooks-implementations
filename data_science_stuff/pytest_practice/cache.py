import time

def should_refresh(last_fetched_at: float, ttl_seconds: int) -> bool:
    """Return True if now - last_fetched_at >= ttl_seconds."""
    return (time.time() - last_fetched_at) >= ttl_seconds
