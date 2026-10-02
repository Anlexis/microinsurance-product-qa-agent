"""AgentCore Platform v1.0"""

# INS-C2-031 — RetrieveNode
# Inner domain node 2: retrieve the most relevant product knowledge-base
# passages for the normalised question.
#
# Deterministic keyword-overlap retrieval over a bundled, in-module knowledge
# base of 少額短期保険 (microinsurance) product terms and 保険業法 disclosure
# rules. An external vector-store or hybrid retriever wires in through
# src/services/service.py without changing this node's contract.
#
# Declared settings consumed (state["runtime_settings"], seeded by the inner
# graph from config/config.yaml):
#   top_k — how many passages the candidate set may hold.
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

# Fallback used when top_k is absent from runtime_settings — i.e. when the
# declared value was absent or failed validation upstream.
_DEFAULT_TOP_K = 5

# Bundled product knowledge base. Each passage carries the keyword tokens used
# for overlap scoring, its source citation, and whether it states an EXCLUSION
# condition. Exclusions are non-suppressible: they surface even when the
# question only asks about coverage.
_KB: List[Dict[str, Any]] = [
    {
        "id": "KB-PET-01",
        "text": (
            "Pet microinsurance (ペット保険) covers veterinary treatment costs "
            "for illness and injury up to the annual policy limit, subject to a "
            "co-payment ratio stated in the policy schedule."
        ),
        "source": "少額短期保険 商品約款 / Pet Plan §3 (coverage scope)",
        "keywords": ["pet", "ペット", "veterinary", "illness", "injury", "coverage", "treatment", "vet", "animal"],
        "is_exclusion": False,
    },
    {
        "id": "KB-PET-EX",
        "text": (
            "Pet microinsurance excludes pre-existing conditions, pregnancy and "
            "birth, preventive care (vaccination, neutering), and treatment during "
            "the initial 30-day waiting period."
        ),
        "source": "少額短期保険 商品約款 / Pet Plan §7 (免責事項 exclusions)",
        "keywords": [
            "pet",
            "ペット",
            "exclusion",
            "免責",
            "pre-existing",
            "waiting",
            "vaccination",
            "pregnancy",
            "preventive",
        ],
        "is_exclusion": True,
    },
    {
        "id": "KB-BIKE-01",
        "text": (
            "Bicycle microinsurance (自転車保険) provides personal liability cover "
            "up to 100 million yen for third-party bodily injury or property damage "
            "caused while riding, plus rider injury benefits."
        ),
        "source": "少額短期保険 商品約款 / Bicycle Plan §2 (liability coverage)",
        "keywords": [
            "bicycle",
            "自転車",
            "liability",
            "third-party",
            "injury",
            "damage",
            "rider",
            "accident",
            "coverage",
        ],
        "is_exclusion": False,
    },
    {
        "id": "KB-BIKE-EX",
        "text": (
            "Bicycle microinsurance does not cover damage from intentional acts, "
            "racing or competition use, riding under the influence of alcohol, or "
            "business/delivery use of the bicycle."
        ),
        "source": "少額短期保険 商品約款 / Bicycle Plan §7 (免責事項 exclusions)",
        "keywords": [
            "bicycle",
            "自転車",
            "exclusion",
            "免責",
            "intentional",
            "racing",
            "alcohol",
            "delivery",
            "business",
        ],
        "is_exclusion": True,
    },
    {
        "id": "KB-PHONE-01",
        "text": (
            "Smartphone microinsurance (スマホ保険) reimburses repair or replacement "
            "cost for accidental screen breakage, water damage, and theft, up to the "
            "device market value with an excess (self-pay) per claim."
        ),
        "source": "少額短期保険 商品約款 / Smartphone Plan §2 (coverage scope)",
        "keywords": [
            "smartphone",
            "スマホ",
            "phone",
            "screen",
            "breakage",
            "water",
            "theft",
            "repair",
            "replacement",
            "coverage",
            "device",
        ],
        "is_exclusion": False,
    },
    {
        "id": "KB-PHONE-EX",
        "text": (
            "Smartphone microinsurance excludes cosmetic damage that does not affect "
            "function, loss without evidence of theft, manufacturer-warranty defects, "
            "and devices older than 5 years at enrolment."
        ),
        "source": "少額短期保険 商品約款 / Smartphone Plan §7 (免責事項 exclusions)",
        "keywords": ["smartphone", "スマホ", "phone", "exclusion", "免責", "cosmetic", "loss", "warranty", "defect"],
        "is_exclusion": True,
    },
    {
        "id": "KB-CLAIM-01",
        "text": (
            "To file a claim, notify the insurer within 30 days of the incident, "
            "submit the claim form, proof of loss (receipts, repair estimate, or "
            "police report), and the policy number. Payout is issued within 30 days "
            "of a complete submission."
        ),
        "source": "少額短期保険 ご請求ガイド / Claims §1 (filing procedure)",
        "keywords": ["claim", "請求", "file", "filing", "notify", "form", "receipt", "payout", "procedure", "how"],
        "is_exclusion": False,
    },
    {
        "id": "KB-PREMIUM-01",
        "text": (
            "Premiums for microinsurance are calculated from the plan type, coverage "
            "amount, insured age band, and (for pet/phone plans) the animal breed or "
            "device model. Premiums are billed monthly and are non-refundable once a "
            "cover month has started."
        ),
        "source": "少額短期保険 重要事項説明書 / Premium §4 (calculation)",
        "keywords": [
            "premium",
            "保険料",
            "calculation",
            "cost",
            "price",
            "monthly",
            "age",
            "coverage",
            "amount",
            "how much",
        ],
        "is_exclusion": False,
    },
    {
        "id": "KB-RENEW-01",
        "text": (
            "Microinsurance policies auto-renew every 12 months unless cancelled "
            "before the renewal date. Under the 保険業法 2025 disclosure rules the "
            "insurer must send renewal terms and any premium changes at least 30 days "
            "in advance."
        ),
        "source": "少額短期保険 重要事項説明書 / Renewal §5 (保険業法 disclosure)",
        "keywords": ["renewal", "renew", "更新", "auto", "cancel", "12 months", "disclosure", "保険業法", "term"],
        "is_exclusion": False,
    },
]


def _score(query_tokens: List[str], passage: Dict[str, Any]) -> float:
    """Keyword-overlap relevance score in [0, 1] for one passage."""
    if not query_tokens:
        return 0.0
    haystack = set(passage["keywords"]) | set(str(passage["text"]).lower().split())
    hits = sum(1 for tok in query_tokens if tok in haystack)
    return hits / len(query_tokens)


class RetrieveNode(FunctionNode):
    """Retrieve the top-k relevant product knowledge-base passages.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        normalized_query: str
        runtime_settings: str  — JSON declared settings; the route by which
                                 config/config.yaml reaches this node.

    Output state keys (partial dict):
        retrieved_passages: str  — JSON-serialised passage list
        retrieval_count:    int
        status:             str
        error_log:          list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        query = state.get("normalized_query") or ""
        if not query.strip():
            logger.error("RetrieveNode: normalized_query is absent from state")
            emit_trace_event(
                "retrieve_failed",
                {"reason": "missing_normalized_query"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: normalized_query is absent from state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RetrieveNode: normalized_query is absent from state"),
            }

        settings: Dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        top_k = int(settings.get("top_k", _DEFAULT_TOP_K))

        query_tokens = query.split()

        # Score every passage (read-only over _KB — never mutate the module list).
        scored = [
            {
                "id": p["id"],
                "text": p["text"],
                "source": p["source"],
                "score": round(_score(query_tokens, p), 4),
                "is_exclusion": p["is_exclusion"],
            }
            for p in _KB
        ]
        scored.sort(key=lambda p: p["score"], reverse=True)
        retrieved = scored[:top_k]

        logger.info(
            "RetrieveNode: query_tokens=%d top_k=%d retrieved=%d",
            len(query_tokens),
            top_k,
            len(retrieved),
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "top_k": top_k,
                "retrieved_count": len(retrieved),
                "top_score": retrieved[0]["score"] if retrieved else 0.0,
            },
            state,
        )

        return {
            "retrieved_passages": to_json(retrieved),
            "retrieval_count": len(retrieved),
            "status": AgentStatus.SUCCESS.value,
        }
