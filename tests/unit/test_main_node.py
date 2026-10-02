# INS-C2-031 — the single-slot reference node, and the trust boundary.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.nodes.main_node import MainNode
from src.nodes.pre_process_node import PreProcessNode


class TestMainNode:
    """The single-slot reference node kept alongside the composed build."""

    def setup_method(self):
        self.node = MainNode()

    def test_success_path(self):
        """Invoked through __call__ rather than execute(), so the trust check
        runs first: MainNode requires ANONYMOUS, so an ANONYMOUS caller clears
        it and execute() runs."""
        state = {
            "validated_input": "test input",
            "node_history": [],
            "error_log": [],
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["result"] is not None

    def test_empty_input(self):
        state = {
            "validated_input": "",
            "node_history": [],
            "error_log": [],
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_execute_method_signature(self):
        """The node contract is execute(self, state) — the framework calls it
        with exactly one argument."""
        import inspect

        assert hasattr(MainNode, "execute"), "MainNode must implement execute()"
        params = list(inspect.signature(MainNode.execute).parameters.keys())
        assert params == ["self", "state"], f"expected (self, state), got {params}"


class TestTrustBoundary:
    """The framework's trust check runs inside __call__, BEFORE execute(), and
    refuses a caller whose trust is below the node's declared level.

    The target is PreProcessNode, the only node requiring VERIFIED_EXTERNAL;
    every other node under src/nodes/ is ANONYMOUS. Invoking via node(state)
    rather than node.execute(state) is what routes through __call__ so the check
    fires — calling execute() directly would bypass it.
    """

    # Lowercase, no address, no digit groups, no consecutive capitalised words,
    # so the framework's own input masking leaves it untouched.
    _PLAIN_QUESTION = "what does my microinsurance policy cover for crop damage"

    def setup_method(self):
        self.node = PreProcessNode()

    def test_pre_process_requires_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_untrusted_caller_is_refused_before_execute(self):
        """The check RETURNS an error mapping rather than raising: the status is
        ERROR, the log carries a trust denial, and execute() never ran — so no
        validated_input was produced."""
        result = self.node(
            {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "user_input": self._PLAIN_QUESTION,
                "correlation_id": "trust-denial",
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any(
            "trust gate denied" in e.lower() for e in result.get("error_log", [])
        ), f"expected a trust denial, got error_log={result.get('error_log')}"
        assert "validated_input" not in result

    def test_trusted_caller_is_admitted(self):
        result = self.node(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "user_input": self._PLAIN_QUESTION,
                "correlation_id": "trust-admit",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None
