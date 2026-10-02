"""AgentCore Platform v1.0"""

# INS-C2-031 — PostProcessNode
# Outer backbone post_process slot: the output boundary.
#
# Two invariants are enforced here, on every answer, before anything reaches the
# caller:
#
#   1. no credential shape appears in the answer;
#   2. the mandatory exclusion notice (免責事項) is present — this template's own
#      stated output contract, which the assembly step is supposed to guarantee
#      and which the boundary verifies independently rather than assuming.
#
# WHY A VIOLATION MUST CLEAR STATE, NOT JUST RAISE
#
# The framework's envelope resolves as `formatted_output or result`, with no
# status check. So a boundary that merely raises — or that returns ERROR while
# leaving an answer-bearing field in place — still ships the un-gated answer
# inside the error envelope. Worse, when the framework's own output scan raises
# (it runs after this node returns, over every value in the returned mapping),
# the node wrapper discards the ENTIRE delta, clearing included. Two properties
# follow, and both are load-bearing:
#
#   - the replacement for formatted_output must be TRUTHY, because a falsy one
#     re-opens the `or result` fallback;
#   - the credential check here must be the framework's own detector, because a
#     narrower local set is a bypass: a value the framework catches and this
#     node misses makes the framework raise and this node's clearing vanish.
#
# The refusal notice names a closed-set reason label and never the matched text.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.service import detect_output_credentials
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# The mandatory exclusion heading the assembly step renders. Its absence means
# the answer did not come through the documented assembly path, so the boundary
# refuses rather than releasing an answer whose disclosure section is missing.
_EXCLUSION_MARKER = "免責事項"

# Every state field that can carry answer text or a payload. A violation blanks
# ALL of them on the same return, so neither the envelope nor a checkpoint
# reader can pick one up. Inert provenance fields (counts, flags) are not in the
# set; the inventory guard below is what stops a future answer-bearing field
# quietly joining them.
_OUTPUT_BEARING_FIELDS: tuple[str, ...] = (
    "answer",
    "generated_answer",
    "reranked_passages",
    "result",
    "formatted_output",
)

# Closed set of reasons a refusal may state. A reason is a LABEL, never the
# matched value and never a message built from the answer.
_REFUSAL_LABELS: Dict[str, str] = {
    "credential_shape": ("The answer was withheld because it contained a credential-shaped value."),
    "missing_exclusion_notice": (
        "The answer was withheld because its mandatory exclusion notice " "(免責事項) was not present."
    ),
    "no_answer_content": "No answer content was produced for this question.",
}


def _cleared_output_state() -> Dict[str, Optional[str]]:
    """Blank every answer-bearing field, as one mapping, in one place."""
    return {field: "" for field in _OUTPUT_BEARING_FIELDS}


def _refusal(reason: str) -> str:
    """Build the caller-facing refusal notice for a closed-set reason label."""
    return (
        f"[ANSWER WITHHELD] {_REFUSAL_LABELS[reason]} "
        "Please contact customer support with the correlation id from this response."
    )


class PostProcessNode(FunctionNode):
    """Enforce the output boundary and publish the final answer.

    Outer backbone post_process slot. Declared ANONYMOUS: the trust boundary is
    PreProcessNode (VERIFIED_EXTERNAL), and re-declaring a higher level here
    would refuse the context the backbone passes through.

    Input state keys:
        answer: str — assembled answer from the inner OutputFormatNode

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        answer:           str        — blanked on a violation
        generated_answer: str        — blanked on a violation
        reranked_passages: str       — blanked on a violation
        status:           str
        error_log:        list[str]  (only on a violation)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        answer = str(state.get("answer") or "")

        # -- No answer to publish --------------------------------------------
        # Reported as a failure, not as a success carrying a stand-in: a caller
        # that receives SUCCESS is entitled to treat the body as an answer. The
        # notice is truthy so the envelope cannot fall back to `result`.
        if not answer.strip():
            logger.error("PostProcessNode: no answer content was produced")
            emit_trace_event(
                "post_process_output_withheld",
                {"reason": "no_answer_content"},
                state,
            )
            notice = _refusal("no_answer_content")
            return {
                **_cleared_output_state(),
                "formatted_output": notice,
                "result": notice,
                "status": AgentStatus.ERROR.value,
                "error_log": ["PostProcessNode: no answer content was produced"],
            }

        # -- Invariant 1: no credential shape --------------------------------
        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=answer,
            domain="INS MicroinsuranceProductQAAgent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(answer, str) and not detect_output_credentials(answer + _review):
            answer = answer + _review

        violation = detect_output_credentials(answer)
        if violation:
            # The label, never the matched text.
            logger.error("PostProcessNode: answer withheld — credential shape (%s)", violation)
            emit_trace_event(
                "post_process_output_withheld",
                {"reason": "credential_shape", "shape": violation},
                state,
            )
            notice = _refusal("credential_shape")
            return {
                **_cleared_output_state(),
                "formatted_output": notice,
                "result": notice,
                "status": AgentStatus.ERROR.value,
                "error_log": ["PostProcessNode: answer withheld — credential shape detected"],
            }

        # -- Invariant 2: the exclusion notice is present --------------------
        if _EXCLUSION_MARKER not in answer:
            logger.error("PostProcessNode: answer withheld — exclusion notice absent")
            emit_trace_event(
                "post_process_output_withheld",
                {"reason": "missing_exclusion_notice"},
                state,
            )
            notice = _refusal("missing_exclusion_notice")
            return {
                **_cleared_output_state(),
                "formatted_output": notice,
                "result": notice,
                "status": AgentStatus.ERROR.value,
                "error_log": ["PostProcessNode: answer withheld — exclusion notice absent"],
            }

        logger.info("PostProcessNode: answer released length=%d", len(answer))
        emit_trace_event(
            "post_process_complete",
            {"output_length": len(answer)},
            state,
        )

        return {
            "formatted_output": answer,
            "result": answer,
            "status": AgentStatus.SUCCESS.value,
        }
