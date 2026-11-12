import ipaddress

from starlette.requests import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_client_ip(request: Request) -> str:
    """
    Resolve the client IP for rate limiting.

    Render routes public traffic through Cloudflare, which supplies
    CF-Connecting-IP with the original client IP.

    For local development, fall back to the direct connection address.
    Never trust X-Forwarded-For directly.
    """
    cf_connecting_ip = request.headers.get("cf-connecting-ip")

    if cf_connecting_ip:
        try:
            return str(ipaddress.ip_address(cf_connecting_ip.strip()))
        except ValueError:
            # Reject malformed or multi-value header contents.
            pass

    return get_remote_address(request)


limiter = Limiter(
    key_func=get_client_ip,
    default_limits=[],
)
