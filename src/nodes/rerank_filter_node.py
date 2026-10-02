"""AgentCore Platform v1.0"""

# INS-C2-031 — RerankFilterNode
# Inner domain node 3: rerank the retrieved passages and drop the low-relevance
# ones, while ALWAYS retaining exclusion passages. Exclusion conditions
# (免責事項) are non-suppressible: they must surface even when the question only
# asks whether something is covered.
#
# Deterministic rerank by retrieval score plus a relevance floor. A
# cross-encoder reranker wires in without changing this node's contract.
#
# Declared settings consumed (state["runtime_settings"]):
#   score_threshold — the relevance floor a coverage passage must clear.
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

logger = logging.getLogger(__name__)

# Fallback relevance floor, used when score_threshold is absent from
# runtime_settings — i.e. when the declared value was absent or failed
# validation upstream.
_DEFAULT_SCORE_THRESHOLD = 0.15

# Cap the number of grounding passages handed to the answer generator.
_MAX_KEPT = 4


class RerankFilterNode(FunctionNode):
    """Rerank + score-filter retrieved passages (exclusion-preserving).

    Sorts passages by retrieval score, drops those below the relevance floor,
    and caps the kept set — but NEVER drops an exclusion passage, and always
    keeps at least the single top passage so an answer is possible.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        retrieved_passages: str  — JSON-serialised passage list
        runtime_settings:   str  — JSON declared settings

    Output state keys (partial dict):
        reranked_passages: str  — JSON-serialised filtered passage list
        status:            str
        error_log:         list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        passages: List[Dict[str, Any]] = from_json(state.get("retrieved_passages"), [])

        if not passages:
            logger.error("RerankFilterNode: retrieved_passages is absent from state")
            emit_trace_event(
                "rerank_filter_failed",
                {"reason": "missing_retrieved_passages"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RerankFilterNode: retrieved_passages is absent from state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RerankFilterNode: retrieved_passages is absent from state"),
            }

        settings: Dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        threshold = float(settings.get("score_threshold", _DEFAULT_SCORE_THRESHOLD))

        # Rerank by score (stable); build a local ordered copy (never mutate input).
        ranked = sorted(passages, key=lambda p: p.get("score", 0.0), reverse=True)

        # Keep above-threshold passages, capped; ALWAYS keep exclusion passages.
        kept: List[Dict[str, Any]] = []
        exclusions: List[Dict[str, Any]] = []
        for p in ranked:
            if p.get("is_exclusion"):
                exclusions.append(p)
            elif p.get("score", 0.0) >= threshold and len(kept) < _MAX_KEPT:
                kept.append(p)

        # Guarantee a non-empty grounding set: fall back to the top passage.
        if not kept and not exclusions:
            kept = ranked[:1]

        # Exclusions always appended (non-suppressible), after the kept coverage set.
        reranked = kept + exclusions

        logger.info(
            "RerankFilterNode: in=%d threshold=%.2f kept=%d exclusions=%d",
            len(passages),
            threshold,
            len(kept),
            len(exclusions),
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "input_count": len(passages),
                "threshold": threshold,
                "kept_count": len(kept),
                "exclusion_count": len(exclusions),
            },
            state,
        )

        return {
            "reranked_passages": to_json(reranked),
            "status": AgentStatus.SUCCESS.value,
        }
