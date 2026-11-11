from slowapi import Limiter
from slowapi.util import get_remote_address

# Use the request's resolved client IP as the rate-limit key.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
)
