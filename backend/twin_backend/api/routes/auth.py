"""GET /auth/me: who the presented API key belongs to (the dashboard shows the role)."""

from __future__ import annotations

from fastapi import APIRouter

from twin_backend.api.deps import Viewer
from twin_backend.api.models import AuthMeResponse

router = APIRouter(tags=["auth"])


@router.get(
    "/auth/me",
    response_model=AuthMeResponse,
    summary="The principal and role of the presented key (anonymous viewer without one)",
    responses={401: {"description": "Invalid key, or a key is required"}},
)
async def auth_me(principal: Viewer) -> AuthMeResponse:
    return AuthMeResponse(
        principal=principal.name, role=principal.role, anonymous=principal.anonymous
    )
