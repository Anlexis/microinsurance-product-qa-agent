"""AgentCore Platform v1.0"""

# INS-C2-031 — InputValidateNode
# Inner domain node 1: domain validation and retrieval normalisation of the
# microinsurance question.
#
# Distinct from PreProcessNode, which owns the trust boundary and the caller
# contract. This node applies the domain rule — a minimum meaningful length —
# and produces the normalized_query that RetrieveNode matches against.
#
# Inner node: ANONYMOUS trust. The trust boundary is PreProcessNode; an inner
# node declaring a higher level would refuse the invocation context that the
# subgraph boundary passes through.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

logger = logging.getLogger(__name__)

# A question shorter than this (after normalisation) is not answerable.
_MIN_QUESTION_CHARS = 2

# Strip characters that are noise for keyword retrieval, keeping alphanumerics,
# whitespace and the CJK ranges the product vocabulary is written in.
_NOISE_RE = re.compile(r"[^\w　-ヿ一-鿿\s]", re.UNICODE)


def _normalise_query(raw: str) -> str:
    """Lower-case, drop punctuation noise and collapse whitespace for retrieval."""
    cleaned = _NOISE_RE.sub(" ", raw.lower())
    return " ".join(cleaned.split())


class InputValidateNode(FunctionNode):
    """Domain validation and retrieval normalisation of the question.

    Produces the normalized_query consumed by RetrieveNode.

    Input state keys:
        validated_input: str — screened question from PreProcessNode; falls back
                               to user_input so the node can be exercised alone.

    Output state keys (partial dict):
        normalized_query: str
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")

        if not isinstance(raw, str) or len(raw.strip()) < _MIN_QUESTION_CHARS:
            logger.error("InputValidateNode: question is too short or missing")
            emit_trace_event(
                "input_validate_failed",
                {"reason": "question_too_short"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: question is too short or missing"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("InputValidateNode: question is too short or missing"),
            }

        normalized_query = _normalise_query(raw)
        if not normalized_query:
            emit_trace_event(
                "input_validate_failed",
                {"reason": "empty_after_normalisation"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: question is empty after normalisation"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("InputValidateNode: question is empty after normalisation"),
            }

        logger.info(
            "InputValidateNode: normalized_query tokens=%d",
            len(normalized_query.split()),
        )
        emit_trace_event(
            "input_validate_complete",
            {"token_count": len(normalized_query.split())},
            state,
        )

        return {
            "normalized_query": normalized_query,
            "status": AgentStatus.SUCCESS.value,
        }
