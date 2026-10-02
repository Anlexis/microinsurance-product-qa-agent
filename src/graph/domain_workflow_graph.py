"""AgentCore Platform v1.0"""

# INS-C2-031 — DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the two-layer nested build. It encapsulates the
# microinsurance product retrieval pipeline:
#
#   START
#     -> input_validate   (InputValidateNode)
#     -> retrieve         (RetrieveNode)
#     -> rerank_filter    (RerankFilterNode)
#     -> generate_answer  (GenerateAnswerNode, grounded)
#     -> output_format    (OutputFormatNode)
#     -> END
#
# Called by ProductQAGraphNode.get_subgraph() (graph.py). get_output() shapes
# the sub_result mapping consumed by merge_output() there.
#
# Contracts enforced here:
#   - inherits BaseGraph (fully custom topology — no forced backbone)
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - initialize / finalize are NOT registered — outer backbone concerns
#   - every inner node declares TrustLevel.ANONYMOUS
#   - every inner node is constructed with no arguments
#   - get_output() is designed together with ProductQAGraphNode.merge_output()
#   - _extra_initial_state() seeds the declared runtime settings into state,
#     which is the only route by which a declared value can reach a node
#   - no platform SDK imports

from typing import Any

from langgraph.graph import END, START

from framework.errors import ConfigError
from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json

# Keys this graph forwards to its nodes. A key outside this set in the inner
# config is a declaration nothing reads, so construction refuses it rather than
# letting it look effective.
_FORWARDED_SETTING_KEYS = frozenset({"top_k", "score_threshold", "system_prompt_template"})


class DomainWorkflowGraph(BaseGraph):
    """Inner retrieval workflow for INS-C2-031.

    Inherits BaseGraph directly for a fully custom node topology. Instantiated
    by ProductQAGraphNode.get_subgraph() in graph.py with the declared,
    already-validated runtime settings.

    Pipeline (linear):
        START
          -> input_validate   (InputValidateNode)
          -> retrieve         (RetrieveNode)
          -> rerank_filter    (RerankFilterNode)
          -> generate_answer  (GenerateAnswerNode, grounded)
          -> output_format    (OutputFormatNode)
          -> END

    All nodes are FunctionNode subclasses with ANONYMOUS trust.
    initialize / finalize are outer backbone concerns and are not registered.
    """

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ins_c2_031_microinsurance_qa_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # -- Config validation ---------------------------------------------------

    def _validate_config(self) -> None:
        """Refuse a settings mapping this graph cannot actually deliver.

        The caller (ProductQAGraphNode._parent_config) has already bounded every
        value it forwards, so what remains to check is the SHAPE: an unknown key
        here would be a declaration that reaches no node, which is the failure
        mode this whole route exists to remove. Failing at construction makes it
        visible instead of silent.
        """
        unknown = sorted(k for k in self.config if k not in _FORWARDED_SETTING_KEYS)
        if unknown:
            raise ConfigError(
                f"[{self.__class__.__name__}] runtime settings not consumed by any node: "
                f"{unknown}. Declare them in config/config.yaml only when a node reads them."
            )

    # -- Initial state -------------------------------------------------------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the declared runtime settings into the inner graph's state.

        The framework calls a node as ``execute(state)`` — one argument — so a
        node cannot be handed a per-invocation config. State is therefore the
        only live route from a declared value to the code that acts on it, and
        seeding it here is what makes config/config.yaml load-bearing rather
        than decorative.

        Stored JSON-serialised: State is a flat TypedDict and checkpoints are
        msgpack-serialised, so a nested mapping is carried as a string.
        """
        return {"runtime_settings": to_json(dict(self.config))}

    # -- Node registration ---------------------------------------------------

    def register_nodes(self) -> None:
        """Register the five domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract. initialize and
        finalize belong to the outer backbone. Every key registered here is
        referenced in add_edges(). Every node is constructed with no arguments.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring ---------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear retrieval topology.

            input_validate -> retrieve -> rerank_filter -> generate_answer
            -> output_format -> END

        There is no conditional branching: every path through this pipeline is
        linear, so route() satisfies the abstract contract but is not wired.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing -------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Satisfy the abstract routing contract.

        The topology is linear and add_conditional_edges() is not used, so this
        method is not reached at run time. It returns END on error so that an
        unexpected invocation cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape --------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the mapping returned to the outer graph as sub_result.

        Received by ProductQAGraphNode.merge_output() in graph.py. Both are
        designed together so the field names cannot drift:

            inner get_output() emits:   "answer", "generated_answer",
                                        "reranked_passages", "retrieval_count",
                                        "status"
            outer merge_output() reads: sub_result.get(...) for each of those.
        """
        return {
            "answer": state.get("answer"),
            "generated_answer": state.get("generated_answer"),
            "reranked_passages": state.get("reranked_passages"),
            "retrieval_count": state.get("retrieval_count", 0),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
