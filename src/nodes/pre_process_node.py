"""AgentCore Platform v1.0"""

# INS-C2-031 — PreProcessNode
# Outer backbone pre_process slot: the trust boundary and the caller contract.
#
# Responsibilities:
#   - require VERIFIED_EXTERNAL caller trust, so an unauthenticated caller is
#     refused here rather than deeper in the pipeline;
#   - refuse an empty or over-long question;
#   - refuse a question carrying a chat-template control token or an explicit
#     instruction-override directive;
#   - mask personal data the framework's own detector cannot see, before the
#     question is written to state and therefore to the checkpoint store;
#   - normalise whitespace and publish validated_input + enriched_context;
#   - emit an audit event for every decision.
#
# Every one of those checks runs INSIDE execute(), not in a gate hook, so a
# direct execute() call proves the behaviour with no framework wrapper in front.
# The framework's own input policy still runs first and blocks some of the same
# payloads; that is defence in depth, not a substitute — measured against the
# installed framework, `<<SYS>>` produces no finding at all there.
#
# The question is plain natural-language text (retrieval intake) — there is no
# structured caller payload to parse.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import mask_personal_data, screen_text

logger = logging.getLogger(__name__)

# Upper bound on a single question. A microinsurance product question is a
# sentence or two; anything longer is not a question this agent can answer.
_MAX_QUESTION_CHARS = 2000


class PreProcessNode(FunctionNode):
    """Trust boundary and caller contract for INS-C2-031.

    The only node requiring VERIFIED_EXTERNAL, so an unauthenticated or
    anonymous caller is refused at the boundary; the inner domain nodes run at
    ANONYMOUS and never see unscreened input.

    Input state keys:
        user_input: str — the caller-supplied natural-language question

    Output state keys (partial dict):
        validated_input:  str        — screened, masked, normalised question
        enriched_context: str        — JSON request metadata (never the question)
        status:           str        — AgentStatus.SUCCESS or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("user_input", "")

        # -- Emptiness -------------------------------------------------------
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: question is empty or missing")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "empty_question"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: question is empty or missing"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("PreProcessNode: question is empty or missing"),
            }

        # -- Size ------------------------------------------------------------
        normalised = " ".join(user_input.split())
        if len(normalised) > _MAX_QUESTION_CHARS:
            logger.warning(
                "PreProcessNode: question too long (%d chars > %d)",
                len(normalised),
                _MAX_QUESTION_CHARS,
            )
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "question_too_long", "length": len(normalised)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: question exceeds {_MAX_QUESTION_CHARS} characters"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"PreProcessNode: question exceeds {_MAX_QUESTION_CHARS} characters"),
            }

        # -- Instruction-override screening ----------------------------------
        # Screened before masking, so a directive cannot be assembled out of
        # what masking replaces. The refusal names the class of the finding and
        # never the text that produced it.
        finding = screen_text(normalised)
        if finding:
            logger.warning("PreProcessNode: question refused — %s", finding)
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "instruction_override", "class": finding},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    f"PreProcessNode: question refused — it carries an instruction-override "
                    f"construct ({finding}). Ask the product question on its own."
                ],
            }

        # -- Personal-data masking -------------------------------------------
        # Applied before the question is written to state, because state is
        # checkpointed. Only the KINDS found are recorded, never the values.
        masked, masked_kinds = mask_personal_data(normalised)

        logger.info(
            "PreProcessNode: question accepted length=%d masked_kinds=%s",
            len(masked),
            masked_kinds,
        )
        emit_trace_event(
            "pre_process_validated",
            {"question_length": len(masked), "masked_kinds": masked_kinds},
            state,
        )

        return {
            "validated_input": masked,
            "enriched_context": to_json(
                {
                    "source": "MicroinsuranceProductQAAgent",
                    "question_length": len(masked),
                    "masked_kinds": masked_kinds,
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
