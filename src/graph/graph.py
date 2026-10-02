"""AgentCore Platform v1.0"""

# INS-C2-031 — outer graph (AgentBaseGraph; two-layer nested retrieval-QA build)
#
# Architecture:
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#              -> finalize -> END
#                    | (RETRY, bounded by max_retry)
#                    +-> pre_process
#
#   The `main` slot is a GraphNode subclass (ProductQAGraphNode) that delegates
#   the whole retrieval workflow to DomainWorkflowGraph (the inner BaseGraph).
#   Domain complexity is encapsulated there; the backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (retrieval topology)
#
# Contracts enforced here:
#   - MicroinsuranceProductQAAgent inherits AgentBaseGraph (framework base class)
#   - super().register_nodes() is called first (fills initialize + finalize)
#   - ProductQAGraphNode occupies self._nodes["main"]
#   - PreProcessNode (VERIFIED_EXTERNAL) occupies the pre_process slot
#   - PostProcessNode (ANONYMOUS) occupies the post_process slot and owns the
#     output boundary
#   - merge_output() returns only changed keys, and deliberately does NOT map
#     the inner `result` into outer state (see its docstring)
#   - the class name matches the `class:` entry point in config/agent.yaml
#   - add_edges() is NOT overridden
#   - no platform SDK imports

from pathlib import Path
from typing import Any, ClassVar, Optional

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_state import AgentState
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.services.service import detect_output_credentials, finite_in_range

# Runtime-parameter file: src/graph/graph.py -> parents[2] is the repository
# root. config/agent.yaml holds only the registration identity (a flat manifest
# with no runtime block); every runtime parameter lives in config/config.yaml,
# which is the file the platform registry loads and passes as Graph(config=...).
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Declared-value bounds. A configuration file is not caller data, but it is
# still an input: an out-of-range or non-finite value here would silently
# disable retrieval or the relevance floor, so each one is parsed and bounded
# exactly like a caller field, and an invalid entry falls back to the consuming
# node's documented default.
_TOP_K_BOUNDS = (1.0, 50.0)
_SCORE_THRESHOLD_BOUNDS = (0.0, 1.0)


def runtime_config() -> dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    The standalone entry point (src/api/server.py) calls this too, so a directly
    deployed agent and a registry-loaded agent see identical configuration.
    Returns an empty mapping — never raises — when the file is absent,
    unreadable, not valid YAML, or not a mapping; the graph then runs on its
    built-in defaults.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


class ProductQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the agent.

    Wraps DomainWorkflowGraph (the inner product-knowledge retrieval workflow).
    Called by the AgentBaseGraph backbone after pre_process and before
    post_process.

    Contracts:
      get_subgraph()  — instantiate and return DomainWorkflowGraph
      extract_input() — pull validated_input from outer state
      merge_output()  — map sub_result fields into the outer state delta
      error_strategy  — "propagate": inner errors re-raise as SubgraphError
    """

    # S-1 declared on the wrapper too: the CI gate only AST-scans FunctionNode
    # subclasses, so a GraphNode main slot passes the pipeline without one and is
    # flagged at review. Same level the nodes in this repo already declare.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported inside the method to avoid a circular
        import at module load time.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates, screens and normalises the raw user_input and
        writes the result to validated_input. Prefer that; fall back to
        user_input only when validated_input is absent.
        """
        return str(state.get("validated_input") or state.get("user_input") or "")

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map the inner graph's sub_result into the outer state delta.

        Returns ONLY changed keys, never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          inner get_output() emits  -> "answer", "generated_answer",
                                       "reranked_passages", "retrieval_count",
                                       "status"
          this merge_output() reads -> sub_result.get(...) for each of those.

        `result` is deliberately NOT mapped here. The framework's envelope
        resolves as ``formatted_output or result``, with no status check, so any
        outer `result` written before the output boundary would be surfaced even
        on an error envelope. Only PostProcessNode — which owns that boundary —
        writes `result`, and it writes it on the same return in which it clears
        everything else. The inner `result` therefore stays inside the subgraph.
        A test pins this: mapping it here re-opens the fallback.
        """
        return {
            "answer": sub_result.get("answer"),
            "generated_answer": sub_result.get("generated_answer"),
            "reranked_passages": sub_result.get("reranked_passages"),
            "retrieval_count": sub_result.get("retrieval_count", 0),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> dict[str, Any]:
        """Forward the declared retrieval settings to the inner graph.

        Reads config/config.yaml (see runtime_config) and returns a flat
        settings mapping. DomainWorkflowGraph seeds it into the inner graph's
        initial state, where the domain nodes read it — the node contract is
        ``execute(self, state) -> dict``, so a node cannot receive a
        per-invocation config argument, and state is the only live route.

        Every value is validated here (type, finiteness, range). A key that is
        absent or invalid is simply not forwarded, and the consuming node falls
        back to its own module default.
        """
        cfg = runtime_config()
        retrieval_raw = cfg.get("retrieval")
        retrieval: dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
        llm_raw = cfg.get("llm")
        llm: dict[str, Any] = llm_raw if isinstance(llm_raw, dict) else {}

        declared: dict[str, Any] = {}

        top_k = finite_in_range(retrieval.get("top_k"), *_TOP_K_BOUNDS)
        if top_k is not None:
            declared["top_k"] = int(top_k)

        score_threshold = finite_in_range(retrieval.get("score_threshold"), *_SCORE_THRESHOLD_BOUNDS)
        if score_threshold is not None:
            declared["score_threshold"] = score_threshold

        template = llm.get("system_prompt_template")
        if isinstance(template, str) and template:
            declared["system_prompt_template"] = template

        return declared


class MicroinsuranceProductQAAgent(AgentBaseGraph):
    """Outer graph for INS-C2-031 — grounded answers about microinsurance
    (少額短期保険) product coverage, exclusions, claims and premiums.

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    ProductQAGraphNode (main slot), which delegates to DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize
        -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode      (VERIFIED_EXTERNAL — trust boundary)
      - main:         ProductQAGraphNode  (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode     (ANONYMOUS — output boundary)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.

    The class name must match the `class:` entry point in config/agent.yaml;
    src/api/server.py imports it as `Graph`.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "MicroinsuranceProductQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all five backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (schema_version, session_id,
        trust_level) and FinalizeNode (response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = ProductQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    def _security_gate_output(self, content: str) -> Optional[str]:
        """Name the first credential shape in a caller-facing answer, else None.

        Declared on the agent class as the template's output-boundary marker.
        The runtime scan is applied inside PostProcessNode.execute(), where a
        violation can also clear the state fields that carry answer text — the
        framework's envelope falls back to ``result`` whatever the status, so
        naming a violation without clearing would not withhold anything.

        Both call the SAME function, so the agent-class marker and the runtime
        boundary can never diverge into two different pattern sets.
        """
        return detect_output_credentials(content or "")

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Entry-point alias: src/api/server.py imports this class as `Graph`. The class
# name MicroinsuranceProductQAAgent is what config/agent.yaml `class:` names.
Graph = MicroinsuranceProductQAAgent
