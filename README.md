# INS-C2-031 — MicroinsuranceProductQAAgent

> **Category**: Cat 2 (domain-specific retrieval pipeline)
> **Industry**: INS (insurance)

## Overview

Answers policyholder questions about Japanese microinsurance (少額短期保険) products — what a
plan covers, what it excludes, how to file a claim, how premiums are calculated, and how renewal
works — over a structured product knowledge base.

Every answer is grounded: the body is assembled from the passages that were actually retrieved
and cleared the relevance floor, each answer cites its sources, and when nothing clears the floor
the agent says so and directs the customer to support rather than filling the gap.

The exclusion conditions (免責事項) always surface. A customer asking "is my pet's illness
covered?" will not think to ask which conditions void the cover, and under the 保険業法 disclosure
rules a coverage answer without them is incomplete — so exclusion passages are never dropped by
the relevance filter, the exclusion notice is always rendered, and the output boundary refuses an
answer that reaches it without one.

Caller questions are bounded, screened for instruction-override constructs, and stripped of
personal data before anything is written to state. Answers are checked for credential shapes on
the way out, and a violating answer is withheld rather than trimmed.

Typical users are customer-support and self-service channels for high-volume, low-premium plans,
where the question load is large and the answer has to be traceable to the policy terms.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >= 3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

To serve it, set a caller token and start the entry point:

```bash
INVOKE_AUTH_TOKEN=<caller token> uvicorn src.api.server:app --port 8000
curl -X POST localhost:8000/invoke \
  -H "Authorization: Bearer <caller token>" \
  -H "Content-Type: application/json" \
  -d @deploy/invoke_payload.json
```

`INVOKE_AUTH_TOKEN` is required. The agent's trust boundary admits only an authenticated caller,
so an unconfigured deployment refuses every request and says why, rather than answering each one
with an empty body.

## Architecture

A fixed five-slot backbone wraps a nested retrieval workflow:

```
initialize -> pre_process -> main -> post_process -> finalize
                              |
                              +-- input_validate -> retrieve -> rerank_filter
                                    -> generate_answer -> output_format
```

`pre_process` is the trust boundary and the caller contract; `main` delegates the whole retrieval
workflow to the inner graph; `post_process` is the output boundary.

## Project Structure

```
src/nodes/       the domain nodes: validate, retrieve, rerank, answer, format, and the two boundaries
src/graph/       the outer graph and the inner retrieval workflow
src/services/    the caller contract: screens, personal-data masking, bounded numbers
src/schemas/     the shared state definition
src/api/         the standalone HTTP entry point
tests/           unit tests, the caller contract, and end-to-end boundary tests
config/          the manifest and the runtime parameters
prompts/         the grounding-prompt contract
docs/            design and test documentation
```

`docs/02_design.md` describes the architecture and the security boundaries;
`docs/03_test_spec.md` maps every shipped test to what it proves.

## Customising

1. Replace the knowledge base in `src/nodes/retrieve_node.py` with your own product corpus.
   Each passage needs an `id`, its `text`, a `source` citation, the `keywords` it is matched on,
   and `is_exclusion` — the last of those is what keeps exclusion conditions non-suppressible.
2. Tune `config/config.yaml`: `retrieval.top_k` caps how many passages are considered and
   `retrieval.score_threshold` is the relevance floor below which a coverage passage is not
   treated as grounding. Every value is bounds-checked before it is used, and an invalid entry
   falls back to the consuming node's default.
3. Adjust the personal-data shapes and the override screens in `src/services/service.py` for your
   jurisdiction — they drive the inbound strip and the refusal set.
4. Re-run the test suite. The tests are written against behaviour, not wording, so they should
   keep passing across those changes.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
