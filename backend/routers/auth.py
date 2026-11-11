
# routers/auth.py
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from rate_limit import limiter
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from pwdlib import PasswordHash

import config

router = APIRouter(prefix="/auth", tags=["authentication"])

password_hasher = PasswordHash.recommended()
bearer_scheme = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def create_access_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(minutes=config.JWT_EXPIRE_MINUTES),
        "type": "admin",
    }
    return jwt.encode(
        payload,
        config.JWT_SECRET,
        algorithm=config.JWT_ALGORITHM,
    )


def verify_admin_token(token: str) -> str:
    try:
        payload = jwt.decode(
            token,
            config.JWT_SECRET,
            algorithms=[config.JWT_ALGORITHM],
        )
        if payload.get("type") != "admin":
            raise ValueError("Invalid token type")

        username = payload.get("sub")
        if username != config.ADMIN_USERNAME:
            raise ValueError("Invalid administrator")

        return username
    except (jwt.InvalidTokenError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired admin session.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def admin_login(request: Request, body: LoginRequest):
    if not config.ADMIN_AUTH_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Administrator authentication is not configured.",
        )

    username_ok = body.username == config.ADMIN_USERNAME

    try:
        password_ok = password_hasher.verify(
            body.password,
            config.ADMIN_PASSWORD_HASH,
        )
    except Exception:
        password_ok = False

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    token = create_access_token(body.username)

    return LoginResponse(
        access_token=token,
        expires_in=config.JWT_EXPIRE_MINUTES * 60,
    )


@router.get("/me")
async def admin_me(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        bearer_scheme
    ),
):
    if not config.ADMIN_AUTH_ENABLED:
        raise HTTPException(status_code=503, detail="Admin auth is disabled.")

    if credentials is None:
        raise HTTPException(status_code=401, detail="Admin login required.")

    username = verify_admin_token(credentials.credentials)
    return {"authenticated": True, "username": username}



async def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        bearer_scheme
    ),
) -> None:
    # Preserve local development only when BOTH authentication
    # mechanisms are disabled.
    if not config.AUTH_ENABLED and not config.ADMIN_AUTH_ENABLED:
        return

    # Accept the existing API key when configured.
    if config.AUTH_ENABLED and x_api_key is not None:
        if x_api_key == config.API_KEY:
            return

    # Accept a valid administrator JWT.
    if config.ADMIN_AUTH_ENABLED and credentials is not None:
        verify_admin_token(credentials.credentials)
        return

    raise HTTPException(
        status_code=401,
        detail="Valid admin session or API key required.",
        headers={"WWW-Authenticate": "Bearer"},
    )
