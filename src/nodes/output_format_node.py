"""AgentCore Platform v1.0"""

# INS-C2-031 — OutputFormatNode
# Inner domain node 5, last in the retrieval pipeline. Assembles the final
# customer-facing answer from the grounded generated_answer: the coverage
# answer, the source citations, and the MANDATORY exclusion notice, which is
# non-suppressible and always rendered. This produces the `answer` string the
# outer PostProcessNode holds to the output boundary.
#
# Inner node: ANONYMOUS trust (the trust boundary is PreProcessNode).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72


def _assemble_answer(generated: Dict[str, Any]) -> str:
    """Assemble the customer-facing answer text (answer + citations + exclusions)."""
    answer_body: str = generated.get("answer", "")
    citations: List[str] = generated.get("citations", [])
    exclusions: List[str] = generated.get("exclusions", [])

    lines = [
        _SEPARATOR,
        "MICROINSURANCE PRODUCT Q&A — ANSWER",
        _SEPARATOR,
        "",
        answer_body,
        "",
    ]

    # Mandatory exclusion notice — ALWAYS rendered, even for a coverage query.
    lines += [_SUBSEP, "重要 / IMPORTANT — Exclusion Conditions (免責事項)"]
    if exclusions:
        lines += [f"  - {ex}" for ex in exclusions]
    else:
        lines.append(
            "  - No product-specific exclusion was retrieved for this question. "
            "Always review the full 商品約款 (policy terms) for the complete "
            "list of exclusions before relying on coverage."
        )
    lines.append("")

    # Citations.
    lines += [_SUBSEP, "Sources / 出典"]
    if citations:
        lines += [f"  [{i + 1}] {src}" for i, src in enumerate(citations)]
    else:
        lines.append("  (no source citation available)")
    lines += ["", _SEPARATOR]

    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final customer-facing answer document (inner domain node).

    Reads generated_answer from state, renders the answer with its citations and
    the mandatory exclusion notice, and writes it to `answer` for the outer
    PostProcessNode.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        generated_answer: str  — JSON-serialised answer object

    Output state keys (partial dict):
        answer:  str
        result:  str  — the inner graph's own result; it is deliberately not
                        mapped into outer state, so the caller-facing envelope
                        can only come from the output boundary
        status:  str
        error_log: list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        generated: Dict[str, Any] = from_json(state.get("generated_answer"), {})

        if not generated or not generated.get("answer"):
            logger.error("OutputFormatNode: generated_answer missing in state")
            emit_trace_event(
                "output_format_failed",
                {"reason": "missing_generated_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: generated_answer missing in state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("OutputFormatNode: generated_answer missing in state"),
            }

        answer = _assemble_answer(generated)

        logger.info(
            "OutputFormatNode: answer_chars=%d exclusions=%d grounded=%s",
            len(answer),
            len(generated.get("exclusions", [])),
            generated.get("grounded", False),
        )
        emit_trace_event(
            "output_format_complete",
            {
                "answer_length": len(answer),
                "exclusion_count": len(generated.get("exclusions", [])),
                "citation_count": len(generated.get("citations", [])),
                "grounded": generated.get("grounded", False),
            },
            state,
        )

        return {
            "answer": answer,
            "result": answer,
            "status": AgentStatus.SUCCESS.value,
        }
