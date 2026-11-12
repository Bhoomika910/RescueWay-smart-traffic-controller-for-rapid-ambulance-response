from starlette.requests import Request

from rate_limit import get_client_ip


def make_request(headers=None, client=("192.0.2.10", 12345)):
    headers = headers or {}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [
            (key.lower().encode(), value.encode())
            for key, value in headers.items()
        ],
        "client": client,
        "server": ("testserver", 80),
    }

    return Request(scope)


def test_uses_valid_cloudflare_ipv4():
    request = make_request({
        "CF-Connecting-IP": "203.0.113.25",
    })

    assert get_client_ip(request) == "203.0.113.25"


def test_uses_valid_cloudflare_ipv6():
    request = make_request({
        "CF-Connecting-IP": "2001:db8::1",
    })

    assert get_client_ip(request) == "2001:db8::1"


def test_ignores_x_forwarded_for():
    request = make_request({
        "X-Forwarded-For": "203.0.113.99",
    })

    assert get_client_ip(request) == "192.0.2.10"


def test_cloudflare_ip_takes_precedence_over_x_forwarded_for():
    request = make_request({
        "CF-Connecting-IP": "203.0.113.25",
        "X-Forwarded-For": "203.0.113.99",
    })

    assert get_client_ip(request) == "203.0.113.25"


def test_invalid_cloudflare_ip_falls_back_to_connection_ip():
    request = make_request({
        "CF-Connecting-IP": "not-an-ip",
    })

    assert get_client_ip(request) == "192.0.2.10"


def test_multiple_ip_values_are_rejected():
    request = make_request({
        "CF-Connecting-IP": "203.0.113.25, 203.0.113.26",
    })

    assert get_client_ip(request) == "192.0.2.10"
