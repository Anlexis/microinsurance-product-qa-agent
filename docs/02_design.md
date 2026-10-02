# Design — INS-C2-031 MicroinsuranceProductQAAgent

## Position in the architecture

- **Agent class**: `MicroinsuranceProductQAAgent`
- **L1 Base (framework base class)**: `AgentBaseGraph` — direct framework inheritance
- **Category / pattern**: Cat 2 · retrieval question-answering (retrieve →
  rerank → grounded compose → render)
- **Industry**: INS (少額短期保険 / microinsurance product questions)
- **Three-layer separation**:
  - State: a flat `TypedDict` — never a Pydantic model, which msgpack cannot
    round-trip through the checkpoint store
  - Node: framework inheritance, overriding `execute(self, state) -> dict` only
  - Graph: composition through `register_nodes()`

## Architecture overview

The template uses the two-layer nested shape:

- **Outer graph** (`src/graph/graph.py`, `MicroinsuranceProductQAAgent`) keeps
  the fixed five-slot backbone. The `main` slot is a `GraphNode` subclass
  (`ProductQAGraphNode`) that delegates the whole retrieval workflow inward.
- **Inner graph** (`src/graph/domain_workflow_graph.py`, `DomainWorkflowGraph`)
  runs the domain nodes in sequence.

### Outer backbone

| Node | Responsibility | Input state | Output state | Class |
|------|---------------|-------------|--------------|-------|
| initialize | schema version, session id, caller trust | — | schema_version, session_id | framework default |
| pre_process | trust boundary (VERIFIED_EXTERNAL) + caller contract: size, override screening, personal-data masking | user_input | validated_input, enriched_context | `PreProcessNode` |
| main | delegate to the inner workflow | validated_input | answer, generated_answer, reranked_passages, retrieval_count | `ProductQAGraphNode` |
| post_process | output boundary: credential scan + exclusion-notice invariant | answer | formatted_output, result | `PostProcessNode` |
| finalize | response metadata, total time | — | response_metadata | framework default |

### Inner retrieval workflow

| Node | Responsibility | Input state | Output state |
|------|---------------|-------------|--------------|
| input_validate | domain validation + retrieval normalisation | validated_input | normalized_query |
| retrieve | keyword-overlap retrieval over the bundled knowledge base (`top_k`) | normalized_query, runtime_settings | retrieved_passages, retrieval_count |
| rerank_filter | rerank by score, apply the relevance floor, retain every exclusion | retrieved_passages, runtime_settings | reranked_passages |
| generate_answer | grounded composition; exclusions always surfaced | reranked_passages, runtime_settings | generated_answer |
| output_format | assemble answer + citations + mandatory exclusion notice | generated_answer | answer, result |

### Data flow

```
Outer:  START -> initialize -> pre_process -> main -> {route} -> post_process
        -> finalize -> END
                 | (retry, bounded by max_retry)
                 +-> pre_process

Inner:  START -> input_validate -> retrieve -> rerank_filter
        -> generate_answer -> output_format -> END
```

### State definition

Every dict/list-valued field is stored JSON-serialised as `Optional[str]`,
because checkpoints are msgpack-serialised and a bare mapping does not survive.

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| runtime_settings | NotRequired[Optional[str]] | JSON declared settings, seeded by the inner graph | at retrieve |
| validated_input | NotRequired[Optional[str]] | screened, masked, normalised question | at main |
| enriched_context | NotRequired[Optional[str]] | JSON request metadata; never the question text | no |
| normalized_query | NotRequired[Optional[str]] | retrieval-normalised query | at retrieve |
| retrieved_passages | NotRequired[Optional[str]] | JSON passage list | at rerank |
| retrieval_count | NotRequired[Optional[int]] | passages retrieved | no |
| reranked_passages | NotRequired[Optional[str]] | JSON filtered passage list | at generate |
| generated_answer | NotRequired[Optional[str]] | JSON {answer, citations, exclusions, grounded} | at output_format |
| answer | NotRequired[Optional[str]] | assembled answer | at post_process |
| result | NotRequired[Optional[str]] | caller-facing result; written only by the output boundary | no |

**State constraints:**

- flat `TypedDict` only — primitives and JSON-serialisable values
- no credentials, tokens, or personal data in state: everything here reaches
  the checkpoint store
- no Pydantic models, dataclasses, or arbitrary objects
- `formatted_output` is inherited and never re-declared with a bare type

## How runtime configuration reaches the code

This is the part most worth reading, because the obvious wiring does not work.

The framework calls a node as `execute(state)` — **one argument**. A node
therefore cannot be handed a per-invocation configuration mapping, and a node
signature that declares one has a parameter nothing ever binds. Separately, the
manifest is flat: it carries no runtime block, so a reader looking for one finds
nothing.

The route that does work runs through state:

1. `src/api/server.py` reads `config/config.yaml` and constructs the graph with
   it, so the backbone's own parameters (`max_retry`) take effect.
2. `ProductQAGraphNode._parent_config()` reads the same file, validates each
   value (type, finiteness, range), and forwards only what survives.
3. `DomainWorkflowGraph._extra_initial_state()` seeds that mapping into the
   inner graph's initial state as `runtime_settings`.
4. Each domain node reads its settings from state.

`DomainWorkflowGraph._validate_config()` refuses a settings key no node reads,
so a declaration that would reach nothing fails at construction instead of
looking effective.

The end-to-end consequence is testable and is tested: changing
`retrieval.top_k` and `retrieval.score_threshold` in `config/config.yaml`
changes what the deployed agent returns.

## Generation mode

`generation_mode: deterministic` in the manifest: the answer body is composed
from retrieved passage text and no model is invoked. Consequently
`config/config.yaml` declares `llm.system_prompt_template` — the grounding
contract, resolved and audited at answer time — and deliberately declares no
sampling parameters, because a temperature that configures nothing is worse
than no declaration at all.

## Security posture

- **Trust boundary.** `PreProcessNode` requires VERIFIED_EXTERNAL; every inner
  node is ANONYMOUS, so the invocation context passes through the subgraph
  boundary rather than being refused there. `src/api/server.py` is the
  entry-point authentication boundary and establishes that level from a bearer
  token; with no token configured it refuses rather than admitting a caller the
  pipeline will reject with an empty body.
- **Caller contract.** Enforced inside `PreProcessNode.execute()`, not in a gate
  hook, so calling `execute()` directly proves it holds with no framework
  wrapper in front. It refuses chat-template control tokens as a class
  (`<|…|>`, `[INST]`, `<<SYS>>`) and anchored instruction-override directives,
  screening both the text as received and the text with markup removed. The
  framework's own input policy runs first and blocks some of the same payloads;
  measured against the installed framework, `<<SYS>>` produces no finding there
  at all, which is why the template screens the family itself.
- **Personal data.** The framework's detector anchors its patterns on `\b`,
  which is computed over `\w` — and `\w` includes Kanji and Kana. Japanese is
  written without spaces, so a personal number written the way a policyholder
  writes it sits against a Kanji and the boundary never matches. The template
  masks with explicit character guards instead, so the result does not depend on
  the script the question was written in. Dates, policy limits, section numbers
  and ratios carry no such shape and are byte-identical after masking.
- **Adapter screening.** The request body is checked for credential shapes with
  the framework's own detector before the graph is entered, and `session_id` is
  restricted to an inert identifier. A credential-shaped question cannot
  succeed — the framework's node-level output scan raises the moment the first
  node returns it — so it is refused with a 400 naming the field rather than
  failing opaquely. The request model ignores unknown fields, so no
  caller-supplied key reaches the agent's context channel.
- **Output boundary.** `PostProcessNode` enforces two invariants: no credential
  shape, and the exclusion notice present. The framework's envelope resolves as
  `formatted_output or result` with **no status check**, so a boundary that
  merely raised, or returned an error while leaving an answer-bearing field
  standing, would still ship the un-gated answer inside the error envelope. On a
  violation the node blanks every answer-bearing field and publishes a truthy
  refusal notice naming a closed-set reason — never the matched value. The
  credential check delegates to the framework's own detector: a narrower local
  set would be a bypass, because a value the framework catches and the node
  misses makes the framework raise inside the wrapper, which discards the node's
  whole delta including its clearing.
- **Audit.** One domain event inside every `execute()`, in addition to the
  framework's own node events.
- **Exclusion invariant.** Exclusion passages are never dropped by the relevance
  filter and the exclusion notice is always rendered, even for a pure coverage
  question and even when no exclusion passage was retrieved.

> **Gate behaviour by node type.** A `FunctionNode` subclass gets the
> framework's final input and output gates automatically and may extend them
> only through the `_extra_*` hooks. A `GraphNode` applies a deliberate no-op at
> the subgraph boundary, because the inner nodes have already been gated — which
> is precisely why the outer output boundary must scan what crosses it.

### Composition

- **Pattern**: subgraph — the outer `AgentBaseGraph` wraps an inner `BaseGraph`
- **Target**: `DomainWorkflowGraph` via `ProductQAGraphNode` (main slot)
- **Error strategy**: propagate — an inner failure raises rather than degrading

`ProductQAGraphNode.merge_output()` deliberately does not map the inner `result`
into outer state. Only the output boundary writes `result`, and it writes it on
the same return in which it clears everything else — so no answer text can reach
the envelope by any other route.

## Import isolation

- The template imports no platform SDK module.
- Import targets are `framework/` and `shared/` only.

## Design decisions

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base class | `AgentBaseGraph` | `AutonomousBaseGraph` | **`AgentBaseGraph`** | a deterministic pipeline, not an autonomous loop |
| Composition | standalone | nested subgraph | **nested subgraph** | the multi-step retrieval topology is cleanly encapsulated |
| Config route | node config argument | state seeding | **state seeding** | the framework calls `execute(state)` — a config argument is never bound |
| Output boundary | refuse by raising | refuse and clear | **refuse and clear** | the envelope falls back to `result` regardless of status |
| Credential set | template-local patterns | the framework's detector | **the framework's detector** | a narrower set is a bypass, not a smaller net |
| Exclusion handling | filter by score | always retain | **always retain** | 保険業法 disclosure — exclusions are non-suppressible |
| Numeric output grid | rounded aggregates | not applicable | **not applicable** | the agent renders no computed monetary aggregate; the invariant it enforces is the exclusion notice |
