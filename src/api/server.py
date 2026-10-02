"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
#
# Entry points are adapters only — no business logic here. When the agent runs
# behind the platform gateway, the gateway calls agent.invoke() directly and
# this module is not in the path.
#
# The adapter enforces three things before the graph is entered, so a request
# that cannot succeed is refused readably instead of failing opaquely inside the
# first node:
#
#   - caller authentication, because the agent's own trust boundary requires a
#     VERIFIED_EXTERNAL caller and would otherwise refuse every request with an
#     empty body and no explanation;
#   - a credential screen on the question, using the framework's own detector,
#     because the framework's node-level output scan raises on a
#     credential-shaped value the moment the first node returns it — the request
#     cannot succeed either way, so a 400 naming the field beats an error
#     envelope the caller cannot act on;
#   - an inert-identifier rule on session_id, which travels into correlation and
#     audit records.
#
# A refusal names the FIELD, never the value.

import os
import secrets
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials
from shared.secrets import factory as secrets_factory
from src.graph.graph import Graph, runtime_config
from src.services.service import is_inert_token

app = FastAPI(title="Agent")

# The runtime parameters are read here and handed to the graph, so a directly
# deployed agent and a registry-loaded one behave identically. Without this the
# backbone would silently use the framework's own defaults for every declared
# value, and config/config.yaml would configure nothing.
agent = Graph(config=runtime_config())
agent.compile()
agent.provision_secrets(secrets_factory(namespace="ins-c2-031", agent_name="MicroinsuranceProductQAAgent"))


class InvokeRequest(BaseModel):
    """Request body. Unknown fields are ignored, so no caller-supplied key can
    reach the agent's context channel."""

    input: str
    session_id: str = ""


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)

    # -- Caller authentication -----------------------------------------------
    # Trust established by upstream middleware is never demoted. A caller no
    # middleware vouched for must present the bearer token and then runs at
    # VERIFIED_EXTERNAL. This adapter is the entry-point auth boundary; the
    # token is a deployment credential, not an agent secret, and no invocation
    # context exists yet from which to read one.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if trust is TrustLevel.ANONYMOUS:
        if not expected:
            # Refusing here is the honest answer: the agent's trust boundary
            # requires VERIFIED_EXTERNAL, so admitting an anonymous caller would
            # produce an error envelope with an empty body on every request.
            raise HTTPException(
                status_code=503,
                detail="Caller authentication is not configured on this deployment.",
            )
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of a clean 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — never disclose whether the token was
            # absent, malformed, or simply wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    # -- Caller-field screening ----------------------------------------------
    # 400, not 422: pydantic owns 422 and answers it with a list of error
    # objects, so reusing it would make client handling ambiguous.
    for field, value in (("input", req.input), ("session_id", req.session_id)):
        if value and detect_credentials(value):
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field}' contains a credential-shaped value. Remove it and retry.",
            )
    if req.session_id and not is_inert_token(req.session_id):
        raise HTTPException(
            status_code=400,
            detail="Field 'session_id' must be 1-64 characters of [A-Za-z0-9_-].",
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        # The framework wheel ships no type information, so invoke() resolves to
        # Any; the cast records the envelope shape without weakening the check.
        envelope: dict[str, Any] = agent.invoke(req.input, ctx=ctx)
        return envelope


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "MicroinsuranceProductQAAgent"}
