# Test Specification — INS-C2-031 MicroinsuranceProductQAAgent

## 1. Strategy

- **Agent**: INS-C2-031 — microinsurance (少額短期保険) product question
  answering; a two-layer nested build (outer `AgentBaseGraph` backbone, inner
  `DomainWorkflowGraph`).
- **Coverage target**: ≥ 90% of branches in `src/nodes/` and `src/graph/`.
- **Test types**: unit (per node and per graph contract) · caller contract ·
  proof-of-boundary (framework contracts and the output boundary, driven through
  the real HTTP entry point).
- **Framework provisioning**: `framework` is installed from the package
  registry. Every test imports the real modules; there are no stub nodes.
- **Caller input**: a plain natural-language product question, not a structured
  payload. The outer `PreProcessNode` owns the trust boundary and the caller
  contract; domain normalisation and retrieval live in the inner graph.
- **Audit events**: `emit_trace_event` is patched at the node MODULE level
  (`monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", ...)`), never
  through a `sys.modules` stub, which would break the real `shared` package the
  framework loads at import time.

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | every backbone and domain node, plus outer/inner graph composition |
| `tests/unit/test_caller_contract.py` | override screening, personal-data masking, bounded declared numbers, inert identifiers, credential-detection parity |
| `tests/unit/test_main_node.py` | the single-slot reference node and the trust-gate contract |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | the framework's gates cannot be replaced by a subclass |
| `tests/proof_of_boundary/test_output_boundary.py` | PB-8: the output boundary, configuration liveness and adapter refusals, through the real HTTP entry point |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6: per-node and backbone invoke order, trust gate, payload alignment |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4: import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5: state serialisation and checkpoint safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7: no cross-boundary human-in-the-loop interrupt; skipped with a reason |

### The three domain invariants

1. **Grounding** — the answer body is composed only from retrieved passage text;
   citations trace back to passage `source` fields.
2. **Abstention** — when no coverage passage clears the relevance floor, the
   answer states that plainly (`grounded=False`) instead of inventing coverage.
3. **Non-suppressible exclusions** — exclusion (免責事項) passages are always
   retained by the relevance filter, even at score 0.0; the exclusion notice is
   always rendered, even for a pure coverage question and even when no exclusion
   passage was retrieved; and the output boundary refuses an answer that reaches
   it without one.

### Canonical payload

The question used by the backbone invoke test and by
`deploy/invoke_payload.json` — the two must stay identical, which
`test_invoke_payload_matches_pb6` asserts:

```
What does pet microinsurance (ペット保険) cover for illness and injury, and what conditions are excluded?
```

A full invoke at VERIFIED_EXTERNAL yields `status=success` and an `output`
carrying the report header and the mandatory exclusion notice.

## 2. Framework compliance

| TC-ID | Test | Expected | Where |
|-------|------|----------|-------|
| TC-01 | State is a flat `TypedDict`; domain fields `NotRequired`; no Pydantic or dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Empty / whitespace / over-length question refused at the trust boundary | `status=error`, error_log populated | `TestPreProcessNode` |
| TC-03 | No credential-named field in state | AST scan + gate | `test_state_safety.py` |
| TC-04 | Node contract is `execute(self, state)` | signature is exactly `(self, state)` | `test_execute_signature_is_state_only` |
| TC-05 | One domain audit event inside every `execute()` | ≥1 per node | gate scan |
| TC-06/07 | The framework's input and output gates cannot be overridden | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | Caller trust enforced before `execute()` | ANONYMOUS refused; VERIFIED_EXTERNAL admitted | `TestTrustBoundary` |
| TC-08a | Trust boundary VERIFIED_EXTERNAL; all other nodes ANONYMOUS | asserted per node | `test_trust_level_*` |
| TC-11 | Output boundary refuses a credential shape and clears state | `status=error`, every answer-bearing field blanked | `TestPostProcessNode` |

## 3. Proof of boundary

| PB-ID | Boundary | Expected | Where |
|-------|----------|----------|-------|
| PB-2 | State serialisation | primitives only | `test_state_safety.py` |
| PB-4 | Import isolation | no platform SDK import under `src/` | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named field, no prohibited type | `test_state_safety.py` |
| PB-6 | Per-node invoke order | trust gate → node_start → input gate → `execute()` → output gate → node_complete | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | `status=success`; node_history = Initialize, PreProcess, ProductQAGraphNode, PostProcess, Finalize | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | VERIFIED_EXTERNAL context — never the internal shortcut | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"]` equals the canonical payload | `test_invoke_payload_matches_pb6` |
| PB-7 | Human-in-the-loop propagation | not applicable (`propagate_hitl=False`); skipped with a reason | `test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | Output boundary and configuration liveness, through the real HTTP entry point | see below | `test_output_boundary.py` |

### PB-8 in detail

Every case drives `POST /invoke` on the real application, and every fault is
injected on the DATA path — the knowledge base, or the subgraph boundary — never
on the boundary under test.

| Case | Fault | Expected |
|------|-------|----------|
| Clean-path control | none | the real answer, with the header and the exclusion notice; the boundary appears in `node_history` |
| Input dependence | none | three different questions produce three different answers |
| Credential at the boundary | the subgraph boundary returns an answer carrying a credential shape | `status=error`; no released text, no traceback, no source path; a truthy refusal notice |
| Missing exclusion notice | the subgraph boundary returns an answer without one | `status=error`; the answer text withheld |
| Credential in the corpus | a knowledge-base passage carries one | the value never reaches the caller |
| Credential-shaped question | caller data | 400 naming the field, never the value |
| Hostile `session_id` | caller data | 400; the value not echoed |
| Unknown body field | caller data | ignored; never reaches the context channel |
| Override construct | `<<SYS>>` payload | refused end to end |
| No configured caller token | deployment | 503 that names the cause |
| Declared retrieval values | `top_k`, `score_threshold` changed | the answer changes accordingly |

## 4. Business logic

| BL-ID | Test | Input | Expected |
|-------|------|-------|----------|
| BL-01 | Grounded happy path | the canonical payload | header + exclusion notice present |
| BL-02 | Grounding | coverage + exclusion passages | answer body drawn from passage text; citations trace to `source` |
| BL-03 | Abstention | exclusion-only retrieval | `grounded=False` + the no-coverage statement |
| BL-04 | Exclusion retained below the floor | coverage@0.8 + exclusion@0.0 | the exclusion survives; low-score coverage is dropped |
| BL-05 | Notice always rendered | `exclusions=[]` | heading + fallback line still present |
| BL-06 | Declared `top_k` respected | `top_k=3` | exactly three passages retrieved |
| BL-07 | Kept coverage capped | six coverage passages @0.9 | four kept |
| BL-08 | Relevance fallback | all coverage below the floor, no exclusion | the single top passage kept |
| BL-09 | Citation de-duplication | two passages, one source | one citation, order preserved |
| BL-10 | Declared floor decides grounding | one passage @0.20, floor 0.1 vs 0.9 | grounded true, then false |
| BL-11 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | the five coupled keys map; `result` is not among them |

### Negative and boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty `user_input` | PreProcessNode | `status=error`, "empty" |
| whitespace-only `user_input` | PreProcessNode | `status=error` |
| question > 2000 characters | PreProcessNode | `status=error`, "exceeds" |
| control token or override directive | PreProcessNode | `status=error`, class named, value never echoed |
| legitimate policy language using the same words | PreProcessNode | `status=success` |
| personal number written against Kanji | PreProcessNode | masked before it reaches state |
| question < 2 characters | InputValidateNode | `status=error`, "short" |
| punctuation only | InputValidateNode | `status=error`, "normalisation" |
| missing `normalized_query` | RetrieveNode | `status=error` |
| missing `retrieved_passages` | RerankFilterNode | `status=error` |
| missing `reranked_passages` | GenerateAnswerNode | `status=error` |
| missing / empty `generated_answer` | OutputFormatNode | `status=error` |
| no answer content | PostProcessNode | `status=error` with a truthy notice — not a success carrying a stand-in |
| credential shape in the answer | PostProcessNode | withheld, every answer-bearing field cleared |
| answer without the exclusion notice | PostProcessNode | withheld |
| non-finite or out-of-range declared value | `_parent_config` | dropped; the node keeps its default |

## 5. Promotion thresholds

The retriever is deterministic (keyword overlap over a bundled corpus), so the
promotion gate asserts the structural and safety contract rather than a
probabilistic retrieval score:

- a full invoke of the canonical payload returns `status=success` with a
  non-empty `output`;
- that output carries the report header and the exclusion notice;
- `deploy/invoke_payload.json["input"]` equals the canonical payload;
- retrieval tuning in `config/config.yaml` is `top_k: 5`,
  `score_threshold: 0.15`, and changing either changes the answer;
- when an external vector retriever is wired in, add a faithfulness and
  answer-relevancy threshold (≥ 0.75) here.

## 6. Execution

- `pytest tests/` against the installed framework wheel.
- Node and graph modules are exercised on both success and failure paths.
- PB-7 ships as a documented skip: this template propagates no interrupt.
