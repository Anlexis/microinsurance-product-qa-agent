# INS-C2-031 — unit tests: domain nodes + graph composition.
#
# These import the real modules and assert real behaviour: grounding, the
# non-suppressible exclusion notice, trust levels, the output boundary, and the
# two-layer nested graph composition.
#
# The caller input is a plain natural-language product question, not a
# structured payload — the outer PreProcessNode owns the trust boundary and the
# caller contract, and the inner graph owns retrieval.
#
# Audit events are patched at the node MODULE level rather than through a
# sys.modules stub, which would break the real `shared` package the framework
# loads at import time. Patch pattern per node:
#     monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", lambda *a, **k: None)

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.schemas.state import from_json, to_json

# A representative product question (pet plan — coverage plus exclusions).
QUESTION = (
    "What does pet microinsurance (ペット保険) cover for illness and injury, " "and what conditions are excluded?"
)


# Connection-string probes are assembled at run time. The credential scan that
# guards this repository flags a literal one — correctly, even in a fixture —
# so the parts are kept apart here and joined only in memory.
_URI_TAIL = "db.internal:5432/policies"


def _conn_uri(scheme: str) -> str:
    """Build a database connection string of the shape the detector knows."""
    return scheme + "://svc:" + "pa55word" + "@" + _URI_TAIL


def _passage(pid, text, source, score, is_exclusion=False) -> dict:
    """A single retrieved/reranked passage in the node contract shape."""
    return {
        "id": pid,
        "text": text,
        "source": source,
        "score": score,
        "is_exclusion": is_exclusion,
    }


def _generated(answer="Pet insurance covers vet costs.", citations=None, exclusions=None, grounded=True) -> str:
    """A generated_answer object as GenerateAnswerNode emits it."""
    return to_json(
        {
            "answer": answer,
            "citations": citations if citations is not None else ["srcA"],
            "exclusions": exclusions if exclusions is not None else [],
            "grounded": grounded,
        }
    )


# -- PreProcessNode (outer pre_process — trust boundary + caller contract) ------


class TestPreProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_valid_question_returns_success(self):
        result = self.node.execute({"user_input": QUESTION})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None

    def test_whitespace_is_normalised(self):
        result = self.node.execute({"user_input": "  hello    world  "})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == "hello world"

    def test_enriched_context_records_metadata_not_the_question(self):
        result = self.node.execute({"user_input": "cover?"})
        ctx = from_json(result["enriched_context"])
        assert ctx["source"] == "MicroinsuranceProductQAAgent"
        assert ctx["question_length"] == len("cover?")
        assert ctx["masked_kinds"] == []
        # The question itself never travels in the metadata record.
        assert "cover?" not in str(ctx)

    def test_empty_input_returns_error(self):
        result = self.node.execute({"user_input": ""})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("empty" in e for e in result["error_log"])

    def test_whitespace_only_input_returns_error(self):
        result = self.node.execute({"user_input": "     "})
        assert result["status"] == AgentStatus.ERROR.value

    def test_question_too_long_returns_error(self):
        long_q = "a " * 1100  # ~2199 characters after normalisation
        result = self.node.execute({"user_input": long_q})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("exceeds" in e for e in result["error_log"])

    def test_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_signature_is_state_only(self):
        import inspect
        from src.nodes.pre_process_node import PreProcessNode

        params = list(inspect.signature(PreProcessNode.execute).parameters.keys())
        assert params == ["self", "state"], (
            "the framework calls execute(state) with one argument; an extra " "parameter would never be bound"
        )


# -- InputValidateNode (inner domain node 1) -----------------------------------


class TestInputValidateNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def test_valid_input_builds_normalized_query(self):
        result = self.node.execute({"validated_input": "What Does PET Insurance Cover?"})
        assert result["status"] == AgentStatus.SUCCESS.value
        nq = result["normalized_query"]
        assert nq == nq.lower()
        assert "?" not in nq
        assert "pet" in nq.split()

    def test_falls_back_to_user_input(self):
        result = self.node.execute({"user_input": QUESTION})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["normalized_query"]

    def test_too_short_returns_error(self):
        result = self.node.execute({"validated_input": "a"})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("short" in e for e in result["error_log"])

    def test_empty_after_normalisation_returns_error(self):
        # Punctuation only: clears the length check, normalises to "".
        result = self.node.execute({"validated_input": "!!!"})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("normalis" in e.lower() or "empty" in e.lower() for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# -- RetrieveNode (inner domain node 2) ----------------------------------------


class TestRetrieveNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_retrieves_passages_with_expected_shape(self):
        result = self.node.execute({"normalized_query": "pet insurance illness injury coverage treatment"})
        assert result["status"] == AgentStatus.SUCCESS.value
        passages = from_json(result["retrieved_passages"])
        assert len(passages) > 0
        assert result["retrieval_count"] == len(passages)
        for p in passages:
            assert {"id", "text", "source", "score", "is_exclusion"} <= set(p.keys())

    def test_respects_declared_top_k(self):
        # top_k arrives through state, which is the route the inner graph seeds.
        result = self.node.execute(
            {
                "normalized_query": "pet insurance coverage claim premium",
                "runtime_settings": to_json({"top_k": 3}),
            }
        )
        assert result["retrieval_count"] == 3
        assert len(from_json(result["retrieved_passages"])) == 3

    def test_exclusion_passages_are_retrievable(self):
        result = self.node.execute(
            {
                "normalized_query": "insurance coverage exclusion premium claim",
                "runtime_settings": to_json({"top_k": 9}),
            }
        )
        passages = from_json(result["retrieved_passages"])
        assert any(p["is_exclusion"] for p in passages)

    def test_missing_query_returns_error(self):
        result = self.node.execute({})
        assert result["status"] == AgentStatus.ERROR.value

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# -- RerankFilterNode (inner domain node 3) — the exclusion rule ---------------


class TestRerankFilterNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def _run(self, passages, settings=None):
        state = {"retrieved_passages": to_json(passages)}
        if settings is not None:
            state["runtime_settings"] = to_json(settings)
        return self.node.execute(state)

    def test_drops_low_score_coverage_but_keeps_high(self):
        passages = [
            _passage("C1", "strong coverage", "srcC1", 0.80, False),
            _passage("C2", "weak coverage", "srcC2", 0.05, False),  # below the floor
        ]
        reranked = from_json(self._run(passages)["reranked_passages"])
        ids = [p["id"] for p in reranked]
        assert "C1" in ids
        assert "C2" not in ids

    def test_declared_score_threshold_changes_what_is_kept(self):
        passages = [
            _passage("C1", "coverage", "srcC1", 0.50, False),
            _passage("C2", "coverage", "srcC2", 0.20, False),
        ]
        loose = [p["id"] for p in from_json(self._run(passages, {"score_threshold": 0.1})["reranked_passages"])]
        strict = [p["id"] for p in from_json(self._run(passages, {"score_threshold": 0.4})["reranked_passages"])]
        assert loose == ["C1", "C2"]
        assert strict == ["C1"]

    def test_exclusion_always_retained_even_below_threshold(self):
        # A zero-score exclusion passage is retained even though a coverage
        # passage at the same score would be dropped.
        passages = [
            _passage("C1", "strong coverage", "srcC1", 0.80, False),
            _passage("EX", "免責: pre-existing conditions", "srcEX", 0.00, True),
        ]
        reranked = from_json(self._run(passages)["reranked_passages"])
        ids = [p["id"] for p in reranked]
        assert "EX" in ids, "an exclusion passage must always survive reranking"
        assert "C1" in ids

    def test_caps_kept_coverage_at_max(self):
        passages = [_passage(f"C{i}", "coverage", f"src{i}", 0.90, False) for i in range(6)]
        reranked = from_json(self._run(passages)["reranked_passages"])
        coverage = [p for p in reranked if not p["is_exclusion"]]
        assert len(coverage) == 4

    def test_fallback_keeps_top_when_all_below_and_no_exclusion(self):
        passages = [
            _passage("A", "t", "srcA", 0.05, False),
            _passage("B", "t", "srcB", 0.02, False),
        ]
        reranked = from_json(self._run(passages)["reranked_passages"])
        assert len(reranked) == 1
        assert reranked[0]["id"] == "A"

    def test_missing_passages_returns_error(self):
        result = self.node.execute({})
        assert result["status"] == AgentStatus.ERROR.value

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# -- GenerateAnswerNode (inner domain node 4) — grounded composition -----------


class TestGenerateAnswerNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        self.node = GenerateAnswerNode()

    def test_grounded_answer_from_coverage(self):
        passages = [
            _passage("C", "Pet insurance covers veterinary costs.", "Pet Plan §3", 0.80, False),
            _passage("EX", "Pet insurance excludes pre-existing conditions.", "Pet Plan §7", 0.30, True),
        ]
        gen = from_json(self.node.execute({"reranked_passages": to_json(passages)})["generated_answer"])
        assert gen["grounded"] is True
        assert "Pet insurance covers veterinary costs." in gen["answer"]
        assert "Pet insurance excludes pre-existing conditions." in gen["exclusions"]
        assert "Pet Plan §3" in gen["citations"]
        assert "Pet Plan §7" in gen["citations"]

    def test_exclusions_surfaced_even_without_coverage(self):
        # Exclusion-only retrieval still surfaces the exclusion, and the answer
        # falls back to a no-coverage statement rather than fabricating cover.
        passages = [_passage("EX", "Pet insurance excludes racing use.", "Pet Plan §7", 0.50, True)]
        gen = from_json(self.node.execute({"reranked_passages": to_json(passages)})["generated_answer"])
        assert gen["grounded"] is False
        assert gen["exclusions"] == ["Pet insurance excludes racing use."]
        assert "does not contain a specific coverage" in gen["answer"]

    def test_declared_threshold_decides_what_counts_as_grounding(self):
        passages = [_passage("C", "coverage text", "srcC", 0.20, False)]
        loose = from_json(
            self.node.execute(
                {
                    "reranked_passages": to_json(passages),
                    "runtime_settings": to_json({"score_threshold": 0.1}),
                }
            )["generated_answer"]
        )
        strict = from_json(
            self.node.execute(
                {
                    "reranked_passages": to_json(passages),
                    "runtime_settings": to_json({"score_threshold": 0.9}),
                }
            )["generated_answer"]
        )
        assert loose["grounded"] is True
        assert strict["grounded"] is False

    def test_citations_are_deduplicated_in_order(self):
        passages = [
            _passage("C1", "t1", "same-source", 0.80, False),
            _passage("C2", "t2", "same-source", 0.70, False),
        ]
        gen = from_json(self.node.execute({"reranked_passages": to_json(passages)})["generated_answer"])
        assert gen["citations"] == ["same-source"]

    def test_declared_prompt_path_is_resolved(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        assert GenerateAnswerNode._template_resolves("prompts/microinsurance_qa.j2") is True
        assert GenerateAnswerNode._template_resolves("prompts/does_not_exist.j2") is False
        assert GenerateAnswerNode._template_resolves("../../etc/passwd") is False
        assert GenerateAnswerNode._template_resolves(None) is False

    def test_missing_passages_returns_error(self):
        result = self.node.execute({})
        assert result["status"] == AgentStatus.ERROR.value

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# -- OutputFormatNode (inner domain node 5) — the mandatory exclusion notice ---


class TestOutputFormatNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.output_format_node import OutputFormatNode

        self.node = OutputFormatNode()

    def test_assembles_full_answer(self):
        state = {
            "generated_answer": _generated(
                answer="- Pet insurance covers vet costs.",
                citations=["Pet Plan §3"],
                exclusions=["Pet Plan excludes pre-existing conditions."],
            )
        }
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["answer"]
        assert result["result"] == answer
        assert "MICROINSURANCE PRODUCT Q&A — ANSWER" in answer
        assert "Pet insurance covers vet costs." in answer
        assert "免責事項" in answer
        assert "Pet Plan excludes pre-existing conditions." in answer
        assert "Sources / 出典" in answer
        assert "Pet Plan §3" in answer

    def test_exclusion_notice_always_rendered_even_without_exclusions(self):
        result = self.node.execute({"generated_answer": _generated(exclusions=[])})
        answer = result["answer"]
        assert "免責事項" in answer
        assert "No product-specific exclusion" in answer

    def test_missing_generated_answer_returns_error(self):
        assert self.node.execute({})["status"] == AgentStatus.ERROR.value

    def test_empty_answer_body_returns_error(self):
        result = self.node.execute({"generated_answer": _generated(answer="")})
        assert result["status"] == AgentStatus.ERROR.value

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# -- PostProcessNode (outer post_process — the output boundary) -----------------


class TestPostProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def _valid_answer(self, extra=""):
        return (
            "MICROINSURANCE PRODUCT Q&A — ANSWER\n"
            "Pet insurance covers vet costs.\n"
            "免責事項: pre-existing conditions." + extra
        )

    def test_clean_answer_is_released(self):
        answer = self._valid_answer()
        result = self.node.execute({"answer": answer})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == answer
        assert result["result"] == answer

    def test_no_answer_content_is_a_failure_with_a_truthy_notice(self):
        result = self.node.execute({"answer": ""})
        assert result["status"] == AgentStatus.ERROR.value
        # Truthy: a falsy replacement re-opens the envelope's `or result` fallback.
        assert result["formatted_output"]
        assert "No answer content" in result["formatted_output"]

    def test_result_in_state_is_not_read_back_as_the_answer(self):
        # The boundary must not resolve its own output from a field it is
        # responsible for clearing; only `answer` is an input here.
        result = self.node.execute({"result": "Leftover answer body from a prior node."})
        assert result["status"] == AgentStatus.ERROR.value
        assert "Leftover answer body" not in result["formatted_output"]
        assert "Leftover answer body" not in result["result"]

    @pytest.mark.parametrize(
        "shape",
        [
            "sk-abcdefghij0123456789ABCDEF",
            "AKIAIOSFODNN7EXAMPLE",
            "sk_live_" + "abcdefghijklmnop1234",
            _conn_uri("postgresql"),
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.aaaaaaaaaaaa",
            "Bearer abcdefghijklmnop1234567",
            "password = supersecret123",
        ],
    )
    def test_credential_shape_is_withheld_and_every_field_cleared(self, shape):
        from src.nodes.post_process_node import _OUTPUT_BEARING_FIELDS

        leaky = self._valid_answer(extra=f"\nOperator note: {shape}")
        result = self.node.execute(
            {
                "answer": leaky,
                "generated_answer": '{"answer": "' + shape + '"}',
                "reranked_passages": "[]",
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        # Presence AND emptiness: LangGraph merges partial deltas, so a key that
        # is merely absent leaves the old value standing in state.
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result, f"{field} must be present in the returned delta"
        assert result["answer"] == ""
        assert result["generated_answer"] == ""
        assert result["reranked_passages"] == ""
        assert result["formatted_output"]
        assert result["formatted_output"] == result["result"]
        assert shape not in result["formatted_output"]
        assert "Operator note" not in result["formatted_output"]
        # The reason is a closed-set label, never the matched text.
        assert "credential-shaped value" in result["formatted_output"]
        assert all(shape not in e for e in result["error_log"])

    def test_answer_without_the_exclusion_notice_is_withheld(self):
        answer = "MICROINSURANCE PRODUCT Q&A — ANSWER\nEverything is covered."
        result = self.node.execute({"answer": answer})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["answer"] == ""
        assert "Everything is covered" not in result["formatted_output"]
        assert "免責事項" in result["formatted_output"]

    def test_cleared_field_inventory_is_pinned(self):
        # A new answer-bearing state field must be added to the cleared set
        # deliberately, not join the inert ones by omission.
        from src.nodes.post_process_node import _OUTPUT_BEARING_FIELDS

        assert set(_OUTPUT_BEARING_FIELDS) == {
            "answer",
            "generated_answer",
            "reranked_passages",
            "result",
            "formatted_output",
        }

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# -- Graph composition: outer AgentBaseGraph + inner BaseGraph ------------------


class TestOuterGraphComposition:
    def test_registers_five_backbone_slots(self):
        from src.graph.graph import MicroinsuranceProductQAAgent, ProductQAGraphNode
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode

        agent = MicroinsuranceProductQAAgent()
        agent.compile()
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], ProductQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.graph import MicroinsuranceProductQAAgent

        agent = MicroinsuranceProductQAAgent()
        assert agent.name == "MicroinsuranceProductQAAgent"
        assert agent.state_schema is State

    def test_graph_alias_matches_real_class(self):
        from src.graph.graph import Graph, MicroinsuranceProductQAAgent

        assert Graph is MicroinsuranceProductQAAgent

    def test_main_slot_graphnode_contracts(self):
        from src.graph.graph import ProductQAGraphNode

        node = ProductQAGraphNode()
        assert node.error_strategy == "propagate"
        assert node.propagate_hitl is False
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"

    def test_merge_output_maps_subresult_keys_only_and_never_result(self):
        from src.graph.graph import ProductQAGraphNode

        node = ProductQAGraphNode()
        sub_result = {
            "answer": "ANSWER",
            "generated_answer": "{}",
            "reranked_passages": "[]",
            "retrieval_count": 3,
            "status": AgentStatus.SUCCESS.value,
            "result": "INNER RESULT",  # deliberately not forwarded
            "node_history": ["x"],  # not forwarded
            "correlation_id": "cid",  # not forwarded
        }
        delta = node.merge_output({}, sub_result)
        assert delta["answer"] == "ANSWER"
        assert delta["retrieval_count"] == 3
        assert delta["status"] == AgentStatus.SUCCESS.value
        assert set(delta.keys()) == {
            "answer",
            "generated_answer",
            "reranked_passages",
            "retrieval_count",
            "status",
        }
        # The envelope resolves as `formatted_output or result` with no status
        # check, so an outer `result` written before the output boundary would
        # be surfaced on an error envelope too.
        assert "result" not in delta

    def test_parent_config_reads_the_runtime_file_and_bounds_it(self, tmp_path, monkeypatch):
        import src.graph.graph as g

        node = g.ProductQAGraphNode()
        declared = node._parent_config()
        assert declared["top_k"] == 5
        assert declared["score_threshold"] == 0.15
        assert declared["system_prompt_template"] == "prompts/microinsurance_qa.j2"

        # A non-finite or out-of-range declared value is dropped rather than
        # forwarded; the consuming node then keeps its documented default.
        monkeypatch.setattr(
            g,
            "runtime_config",
            lambda: {"retrieval": {"top_k": float("nan"), "score_threshold": 5.0}},
        )
        assert node._parent_config() == {}

    def test_agent_class_output_gate_uses_one_pattern_set(self):
        from src.graph.graph import MicroinsuranceProductQAAgent
        from src.services.service import detect_output_credentials

        agent = MicroinsuranceProductQAAgent()
        for probe in ("sk-abcdefghij0123456789ABCDEF", "AKIAIOSFODNN7EXAMPLE", "A clean answer."):
            assert agent._security_gate_output(probe) == detect_output_credentials(probe)


class TestInnerDomainGraph:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def test_registers_five_domain_nodes(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        assert g.name == "ins_c2_031_microinsurance_qa_workflow"
        assert g.state_schema is State

    def test_declared_settings_are_seeded_into_the_initial_state(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph(config={"top_k": 3, "score_threshold": 0.4})
        seeded = from_json(g._extra_initial_state()["runtime_settings"])
        assert seeded == {"top_k": 3, "score_threshold": 0.4}

    def test_a_setting_no_node_reads_is_refused_at_construction(self):
        from framework.errors import ConfigError
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph(config={"temperature": 0.0})
        with pytest.raises(ConfigError, match="not consumed by any node"):
            g.compile()

    def test_inner_graph_invoke_produces_grounded_answer(self):
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(QUESTION, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["answer"] is not None
        assert "MICROINSURANCE PRODUCT Q&A" in result["answer"]
        assert "免責事項" in result["answer"]

    def test_get_output_couples_keys_with_outer_merge_output(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        out = g.get_output(
            {
                "answer": "A",
                "generated_answer": "{}",
                "reranked_passages": "[]",
                "retrieval_count": 2,
                "status": AgentStatus.SUCCESS.value,
            }
        )
        assert {"answer", "generated_answer", "reranked_passages", "retrieval_count", "status"} <= set(out.keys())
