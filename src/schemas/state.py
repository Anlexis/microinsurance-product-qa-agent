"""AgentCore Platform v1.0"""

# INS-C2-031 — Microinsurance (少額短期保険) product Q&A state.
#
# State must be a flat TypedDict — never a Pydantic model. Checkpoints are
# msgpack-serialised, and a Pydantic object serialises silently wrong. Extend
# AgentState with agent-specific fields only; never put credentials or secrets
# in state, because everything here reaches the checkpoint store.
#
# Two-layer nested build: outer backbone (AgentBaseGraph) + inner domain
# workflow (BaseGraph). The fields below cover both layers.
#
# Every dict/list-valued field is stored as a JSON-serialised Optional[str].
# Use to_json() / from_json() at every producer and consumer — one contract end
# to end. Typing such a field as a bare dict/list causes msgpack serialisation
# failures at checkpoint time.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialise a value to a JSON string for state storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialise a JSON string held in state."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for INS-C2-031 (microinsurance product Q&A).

    All shared fields (user_input, status, session_id, node_history, error_log,
    hitl_*, formatted_output, ...) are inherited from AgentState and are never
    re-declared here.
    """

    # ------------------------------------------------------------------
    # Runtime settings — seeded by DomainWorkflowGraph._extra_initial_state()
    # ------------------------------------------------------------------

    # JSON-serialised mapping of the declared, already-validated runtime
    # settings (top_k, score_threshold, system_prompt_template). This is the
    # route by which config/config.yaml reaches the domain nodes: the framework
    # calls a node as execute(state), so a node cannot receive a per-invocation
    # config argument.
    runtime_settings: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — written by PreProcessNode (pre_process slot)
    # ------------------------------------------------------------------

    # Trust-validated, screened, personal-data-masked, whitespace-normalised
    # policyholder question. Produced by PreProcessNode; consumed by the inner
    # InputValidateNode.
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised request metadata (source, question length, whether any
    # personal-data kind was masked). Never carries the question text itself.
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # Retrieval-normalised question (lower-cased, de-noised) produced by the
    # inner InputValidateNode and consumed by RetrieveNode.
    normalized_query: NotRequired[Optional[str]]

    # JSON-serialised list of retrieved knowledge-base passages. Each item:
    # {"id": str, "text": str, "source": str, "score": float,
    #  "is_exclusion": bool}
    retrieved_passages: NotRequired[Optional[str]]

    # Number of passages retrieved before reranking and filtering.
    retrieval_count: NotRequired[Optional[int]]

    # JSON-serialised list of reranked and score-filtered passages. Same item
    # shape as retrieved_passages. Exclusion passages are never filtered out —
    # the exclusion notice is non-suppressible.
    reranked_passages: NotRequired[Optional[str]]

    # JSON-serialised grounded-answer object:
    # {"answer": str, "citations": list[str], "exclusions": list[str],
    #  "grounded": bool}
    generated_answer: NotRequired[Optional[str]]

    # Final assembled answer text (answer body, citations, exclusion notice).
    # Built by the inner OutputFormatNode.
    answer: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — written by PostProcessNode (post_process slot)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller. The framework's envelope resolves
    # as `formatted_output or result` with no status check, so this field is
    # written ONLY by the node that owns the output boundary, and only on the
    # same return in which every other answer-bearing field is set.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing — framework-managed; do not write these from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history is inherited from AgentState
