"""
Rate limiting middleware for API protection
"""

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm
    """

    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls  # Number of calls allowed
        self.period = period  # Time period in seconds
        self.clients: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)

        # Check rate limit
        if not self._is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": self.calls,
                    "period": self.period,
                    "retry_after": self._get_retry_after(client_ip),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_calls(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.period)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers first (for proxy/load balancer setups)
        forwarded_ip = request.headers.get("X-Forwarded-For")
        if forwarded_ip:
            return forwarded_ip.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"

    def _is_allowed(self, client_ip: str) -> bool:
        """Check if client is within rate limit"""
        now = time.time()
        window_start = now - self.period

        # Clean old entries
        client_calls = self.clients[client_ip]
        while client_calls and client_calls[0] < window_start:
            client_calls.popleft()

        # Check if within limit
        if len(client_calls) >= self.calls:
            return False

        # Record this call
        client_calls.append(now)
        return True

    def _get_remaining_calls(self, client_ip: str) -> int:
        """Get remaining calls for client"""
        current_calls = len(self.clients[client_ip])
        return max(0, self.calls - current_calls)

    def _get_retry_after(self, client_ip: str) -> int:
        """Get seconds until client can make requests again"""
        client_calls = self.clients[client_ip]
        if not client_calls:
            return 0

        oldest_call = client_calls[0]
        return max(0, int(self.period - (time.time() - oldest_call)))


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP whitelist middleware for production security
    """

    def __init__(self, app, allowed_ips: list = None):
        super().__init__(app)
        self.allowed_ips = set(allowed_ips or [])
        # Add localhost by default
        self.allowed_ips.update(["127.0.0.1", "::1", "localhost"])

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)

        # Skip check for health endpoints
        if request.url.path.startswith("/health"):
            return await call_next(request)

        # Check if IP is allowed (skip in development)
        if self.allowed_ips and client_ip not in self.allowed_ips:
            raise HTTPException(status_code=403, detail=f"Access forbidden for IP: {client_ip}")

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_ip = request.headers.get("X-Forwarded-For")
        if forwarded_ip:
            return forwarded_ip.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"
