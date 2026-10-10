"""Sign in with Google, and the cookie that keeps someone signed in.

The authorisation-code flow, by hand, with no auth library:

    1. /auth/google/login  sends the browser to Google with a signed,
                           single-use `state` value
    2. Google sends it back to /auth/google/callback with a `code`
    3. we swap that code for an access token, server to server
    4. we ask Google who it belongs to
    5. we upsert the user and set a signed cookie

Step 4 is why there is no JWT library here. Google returns an id_token
too, and verifying it properly means fetching JWKS and checking an RS256
signature. Calling the userinfo endpoint with the access token gets the
same claims over a TLS connection straight to Google, with none of that
machinery. The trade is one extra HTTP request per sign-in, which
happens once per fortnight per student.

No passwords are stored, and no TOTP secrets: Google owns the password
and the 2-step verification. The only secret here is the key that signs
the session cookie.

Scopes are openid, email and profile - the three baseline sign-in
scopes. They are non-sensitive, so this needs no Google verification
review and costs nothing. Adding any Gmail or Drive scope changes that,
and would drag in brand verification or a paid annual assessment.
"""

from __future__ import annotations

import logging
import secrets
from datetime import date

import httpx
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from config import settings
from database.models import User
from database.store import get_user, upsert_google_user

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

SCOPES = "openid email profile"

_STATE_COOKIE = "aika_oauth_state"
_STATE_MAX_AGE = 600  # ten minutes to finish signing in

# A random key when none is configured. This logs everyone out on every
# restart, which is the safe failure - a hardcoded default would let
# anyone forge a session cookie for any user.
_SECRET = settings.SESSION_SECRET or secrets.token_urlsafe(48)
if not settings.SESSION_SECRET:
    logger.warning(
        "[auth] SESSION_SECRET is not set - sessions will not survive a restart"
    )

_session_signer = URLSafeTimedSerializer(_SECRET, salt="aika-session")
_state_signer = URLSafeTimedSerializer(_SECRET, salt="aika-oauth-state")


class AuthNotConfigured(RuntimeError):
    """Raised when the Google client ID or secret is missing."""


def is_configured() -> bool:
    return bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)


def callback_host() -> str:
    """The host:port Google will send the browser back to.

    Cookies are scoped by host, and a browser treats 127.0.0.1 and
    localhost as different hosts even though they are the same machine.
    Start the login on one and finish on the other and the state cookie
    is never sent back, so the callback can only report invalid_state -
    which reads like a CSRF attack rather than a typo in the address
    bar. /auth/google/login uses this to bounce to the right host first.
    """
    return urlsplit(settings.OAUTH_REDIRECT_URI).netloc


def _require_configured() -> None:
    if not is_configured():
        raise AuthNotConfigured(
            "Google sign-in is not configured. Set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET in .env - see the README."
        )


# --- step 1: send the browser to Google -------------------------------

def build_login_url() -> tuple[str, str]:
    """The URL to redirect to, and the signed state to set as a cookie.

    `state` is what stops a cross-site request forgery on the callback:
    without it, an attacker could hand someone a callback URL carrying
    their own code and silently sign the victim into the attacker's
    account. The nonce is signed rather than stored, so nothing has to
    be remembered between the two requests.
    """
    _require_configured()

    nonce = secrets.token_urlsafe(24)
    state = _state_signer.dumps(nonce)

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        # Always show the account chooser: shared lab machines are the
        # normal case here, and silently reusing whoever signed in last
        # would put one student's history in front of another.
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{httpx.QueryParams(params)}", state


def verify_state(returned: str | None, from_cookie: str | None) -> bool:
    """The state must come back unchanged, and match the cookie we set."""
    if not returned or not from_cookie or returned != from_cookie:
        return False
    try:
        _state_signer.loads(returned, max_age=_STATE_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return False
    return True


# --- steps 3 and 4: code -> token -> identity -------------------------

async def exchange_code_for_user(code: str) -> User:
    """Swap the one-time code for an access token, then ask Google who
    it belongs to, and return the matching row."""
    _require_configured()

    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.OAUTH_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        if token_response.status_code != 200:
            logger.warning("[auth] token exchange failed: %s", token_response.text[:200])
            raise HTTPException(status_code=400, detail="Google sign-in failed.")

        access_token = token_response.json().get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="Google sign-in failed.")

        info_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if info_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not read Google profile.")

    info = info_response.json()
    sub, email = info.get("sub"), info.get("email")
    if not sub or not email:
        raise HTTPException(status_code=400, detail="Google returned an incomplete profile.")

    # An unverified address means someone typed it, not that Google
    # confirmed it - two accounts could claim the same one.
    if not info.get("email_verified", False):
        raise HTTPException(
            status_code=403,
            detail="Please verify your email address with Google first.",
        )

    return upsert_google_user(sub=sub, email=email, email_verified=True)


# --- the login cookie -------------------------------------------------

def issue_session(response: Response, user_id: int) -> None:
    token = _session_signer.dumps({"uid": user_id})
    response.set_cookie(
        settings.SESSION_COOKIE,
        token,
        max_age=settings.SESSION_MAX_AGE,
        # httponly: JavaScript cannot read it, so an XSS bug in the chat
        # UI cannot walk off with someone's session.
        httponly=True,
        # lax: sent on normal navigation, withheld on cross-site POSTs.
        samesite="lax",
        secure=settings.COOKIE_SECURE,
        path="/",
    )


def clear_session(response: Response) -> None:
    response.delete_cookie(settings.SESSION_COOKIE, path="/")


def set_state_cookie(response: Response, state: str) -> None:
    response.set_cookie(
        _STATE_COOKIE,
        state,
        max_age=_STATE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=settings.COOKIE_SECURE,
        path="/",
    )


def clear_state_cookie(response: Response) -> None:
    response.delete_cookie(_STATE_COOKIE, path="/")


def current_user(request: Request) -> User | None:
    """Whoever the cookie says is signed in, or None.

    The user is re-read from the database on every request rather than
    trusted from the cookie. The cookie carries only an id, so a user
    deleted or changed since signing in is picked up immediately.
    """
    token = request.cookies.get(settings.SESSION_COOKIE)
    if not token:
        return None
    try:
        data = _session_signer.loads(token, max_age=settings.SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    user_id = data.get("uid") if isinstance(data, dict) else None
    return get_user(user_id) if user_id else None


def require_user(request: Request) -> User:
    """FastAPI dependency: 401 unless signed in."""
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return user


def parse_dob(raw: str | None) -> date | None:
    """An ISO date from the onboarding form, or None.

    Rejects impossible values rather than storing them: a typo in a year
    becomes a wrong age in the NGO's reporting, which nobody would catch.
    """
    if not raw:
        return None
    try:
        value = date.fromisoformat(raw)
    except ValueError:
        raise HTTPException(status_code=422, detail="Date of birth must be YYYY-MM-DD.")

    today = date.today()
    if value > today:
        raise HTTPException(status_code=422, detail="Date of birth is in the future.")
    if value.year < today.year - 120:
        raise HTTPException(status_code=422, detail="Date of birth is not plausible.")
    return value


def public_user(user: User) -> dict:
    """What the browser is allowed to know about the signed-in user."""
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "date_of_birth": user.date_of_birth.isoformat() if user.date_of_birth else None,
        "onboarded": user.onboarded,
        "role": user.role,
    }
