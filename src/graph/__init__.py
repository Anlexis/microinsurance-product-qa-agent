"""AgentCore Platform v1.0"""

# Entry-point export: the registry imports the agent class from this package.
# config/agent.yaml declares `module: src.graph` + `class: MicroinsuranceProductQAAgent`,
# so `import src.graph` MUST expose the class for AgentRegistry to resolve the
# manifest entry point. Re-export it (and the backward-compat `Graph` alias that
# src/api/server.py imports) from the package root.
from .graph import Graph, MicroinsuranceProductQAAgent

__all__ = ["Graph", "MicroinsuranceProductQAAgent"]
