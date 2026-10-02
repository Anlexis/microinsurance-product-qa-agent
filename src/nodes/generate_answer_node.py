"""AgentCore Platform v1.0"""

# INS-C2-031 — GenerateAnswerNode
# Inner domain node 4: compose a GROUNDED answer strictly from the reranked
# passages.
#
# The answer body is assembled only from retrieved passage text — no free-form
# generation — and exclusion conditions are always surfaced. The manifest
# declares generation_mode: deterministic, meaning no model is invoked; the
# declared grounding prompt (llm.system_prompt_template) states the contract a
# model-backed build must hold to, and this node resolves it and records
# whether it resolved, so a configuration naming a file that is not there is
# visible in the audit trail rather than silent.
#
# Grounding honesty: a coverage passage is reported as grounding only when its
# retrieval score clears the relevance floor. Below-floor coverage — the rerank
# fallback singleton, for instance — is not synthesised into the answer body and
# does not set grounded=True, so a low-confidence retrieval never looks grounded.
#
# Declared settings consumed (state["runtime_settings"]):
#   score_threshold        — the relevance floor for grounding
#   system_prompt_template — repository-relative path to the grounding prompt
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

from src.schemas.state import from_json, to_json
from src.services.service import resolve_repo_path

logger = logging.getLogger(__name__)

# Fallback relevance floor, used when score_threshold is absent from
# runtime_settings. Kept equal to RerankFilterNode's fallback so the two nodes
# agree when neither receives a declared value.
_DEFAULT_SCORE_THRESHOLD = 0.15

# Appended when no coverage passage cleared the relevance floor.
_NO_COVERAGE_MSG = (
    "The knowledge base does not contain a specific coverage answer for this "
    "question. Please contact customer support for a definitive determination."
)


def _compose_answer(coverage: List[Dict[str, Any]]) -> str:
    """Compose the grounded coverage answer body from coverage passages."""
    if not coverage:
        return _NO_COVERAGE_MSG
    return "\n".join(f"- {p['text']}" for p in coverage)


class GenerateAnswerNode(FunctionNode):
    """Compose a grounded microinsurance answer from the reranked passages.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        reranked_passages: str  — JSON-serialised filtered passage list
        runtime_settings:  str  — JSON declared settings

    Output state keys (partial dict):
        generated_answer: str  — JSON-serialised answer object:
            {answer: str, citations: list[str], exclusions: list[str],
             grounded: bool}
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        passages: List[Dict[str, Any]] = from_json(state.get("reranked_passages"), [])

        settings: Dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        threshold = float(settings.get("score_threshold", _DEFAULT_SCORE_THRESHOLD))
        template_resolved = self._template_resolves(settings.get("system_prompt_template"))

        if not passages:
            logger.error("GenerateAnswerNode: reranked_passages is absent from state")
            emit_trace_event(
                "generate_answer_failed",
                {"reason": "missing_reranked_passages"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["GenerateAnswerNode: reranked_passages is absent from state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("GenerateAnswerNode: reranked_passages is absent from state"),
            }

        coverage = [p for p in passages if not p.get("is_exclusion")]
        exclusion_passages = [p for p in passages if p.get("is_exclusion")]

        grounded_coverage = [p for p in coverage if float(p.get("score", 0.0)) >= threshold]

        answer_body = _compose_answer(grounded_coverage)
        exclusions = [p["text"] for p in exclusion_passages]
        # Citations name only the passages that inform the answer — the grounded
        # coverage plus the always-surfaced exclusions. Order preserved,
        # de-duplicated by source.
        citations: List[str] = []
        for p in grounded_coverage + exclusion_passages:
            src = p.get("source", "")
            if src and src not in citations:
                citations.append(src)

        grounded = bool(grounded_coverage)
        generated = {
            "answer": answer_body,
            "citations": citations,
            "exclusions": exclusions,
            "grounded": grounded,
        }

        logger.info(
            "GenerateAnswerNode: coverage=%d grounded_coverage=%d exclusions=%d "
            "citations=%d threshold=%.2f grounded=%s",
            len(coverage),
            len(grounded_coverage),
            len(exclusions),
            len(citations),
            threshold,
            grounded,
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "coverage_count": len(coverage),
                "grounded_coverage_count": len(grounded_coverage),
                "exclusion_count": len(exclusion_passages),
                "citation_count": len(citations),
                "threshold": threshold,
                "grounded": grounded,
                "prompt_template_resolved": template_resolved,
            },
            state,
        )

        return {
            "generated_answer": to_json(generated),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _template_resolves(declared: Any) -> bool:
        """True when the declared grounding-prompt path names a file that exists."""
        candidate = resolve_repo_path(declared)
        return candidate is not None and candidate.is_file()
