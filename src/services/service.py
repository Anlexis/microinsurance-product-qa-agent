"""AgentCore Platform v1.0"""

# Service layer for INS-C2-031 — shared, node-independent validation of
# caller-supplied data, plus the domain seam for a real product knowledge base.
#
# No business logic, no routing, no credentials. The HTTP adapter and the domain
# nodes call the same helpers, so both entry paths enforce one contract.
#
# Four rules are implemented here, and every caller-facing check in this
# template is one of them:
#
#   1. inert identifiers — a caller string that travels into correlation or
#      audit records is restricted to a closed alphabet, so caller text can
#      never read as structure or as an instruction;
#   2. finite bounded numbers — every declared number the agent acts on is
#      parsed to a real, finite value inside an explicit range. NaN is the
#      dangerous case: it parses through float() and compares False against
#      every bound, so an unchecked score_threshold would silently keep every
#      retrieved passage — or drop every one — with no error surfacing;
#   3. instruction-override screening — caller text is refused when it carries
#      a chat-template control token or an explicit directive to discard the
#      agent's own instructions. Screened raw AND with markup removed, because
#      a directive can be spliced ("ig<b>nore ...") so that only one of the two
#      forms is readable;
#   4. personal-data masking that does not depend on word boundaries. The
#      framework's own detector anchors its patterns with \\b, which is computed
#      over \\w — and \\w includes Kanji and Kana. Japanese is written without
#      spaces, so a personal number written the way a policyholder writes it
#      ("個人番号1234-5678-9012を確認") sits directly against a Kanji and the
#      boundary never matches. Measured against the installed framework: the
#      same digits between ASCII spaces ARE masked and the adjacent form is not.
#      The patterns below use explicit character guards instead, so the result
#      does not depend on the script the question was written in. Dates,
#      section numbers, policy limits and ratios carry no such shape and stay
#      byte-identical.
#
# The template owns these guarantees; the framework's own input policy runs in
# front of the nodes but classifies some control-token forms as audit-only, and
# one form it does not score at all. Every check here is enforced inside the
# node that owns the caller contract, and is proved by calling execute()
# directly with no framework wrapper in front.

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

from framework.security.credential_detector import detect_credentials

# src/services/service.py -> parents[2] is the repository root. A declared
# repository-relative path is resolved against it and must stay inside it.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Caller strings that travel into correlation and audit records are restricted
# to a closed alphabet — free text there is caller-controlled record injection.
_INERT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# A field NAME is caller data too. Only an already-inert name is ever echoed in
# a refusal message; anything else is reported positionally.
_SAFE_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")

# Chat-template control tokens, screened as a CLASS rather than as a list of
# known strings: any `<|...|>` marker, the instruction brackets used by several
# instruction-tuned model families, and the system-block markers. None of these
# carries meaning in a microinsurance question, so refusing the whole family
# costs nothing — and a phrase-only screen misses every one of them. The
# system-block marker matters in particular: the installed framework's own
# detector returns no finding at all for `<<SYS>>`, while it scores `[INST]`
# and `<|im_start|>` as high-confidence and blocks them.
_CONTROL_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chat_control_token", re.compile(r"<\|[^|>]{0,64}\|>")),
    ("instruction_bracket", re.compile(r"\[/?INST]", re.IGNORECASE)),
    ("system_block_marker", re.compile(r"<</?SYS>>", re.IGNORECASE)),
)

# Explicit instruction-override directives. Every pattern is anchored on both
# sides of a full verb+object phrase: a bare verb ("exclude", "override") is NOT
# enough, because ordinary policy language contains those words ("what
# conditions are excluded?", "override the co-payment ratio") and a screen that
# refuses real questions is worse than no screen at all. Probed against this
# template's own knowledge-base text and its canonical question — see the
# caller-contract test module.
_DIRECTIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?<![A-Za-z])(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|any\s+|the\s+|your\s+|these\s+)*"
            r"(?:previous|prior|above|preceding|earlier|foregoing|system)?\s*"
            r"(?:instructions?|rules?|prompts?|directives?|guidelines?|constraints?)"
            r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"(?<![A-Za-z])(?:reveal|disclose|repeat|print|output|dump|show)\s+"
            r"(?:me\s+|us\s+)?(?:your|the)\s+(?:full\s+|entire\s+|original\s+)?"
            r"(?:system\s+prompt|system\s+message|initial\s+instructions)"
            r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        # A role RE-ASSIGNMENT, not any sentence starting "you are now". The
        # article is what separates the two: "you are now an underwriter"
        # asserts a new identity, while "you are now handling my claim" — which
        # a policyholder writes — is a gerund and must pass.
        "persona_override",
        re.compile(
            r"(?<![A-Za-z])you\s+are\s+(?:now|no\s+longer)\s+(?:an?|the)\s+"
            r"|(?<![A-Za-z])you\s+are\s+no\s+longer\s+"
            r"(?:bound|required|restricted|limited|allowed|obliged)(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_injection",
        re.compile(
            r"(?<![A-Za-z])new\s+(?:instructions?|rules?|system\s+prompt)\s*[:：]",
            re.IGNORECASE,
        ),
    ),
)

# Markup and invisible characters are removed before the SECOND screening pass,
# so a directive spliced across inline tags is readable to the same patterns.
_MARKUP_TAG_RE = re.compile(r"<[^<>]{0,64}>")
_INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")

_MASK = "[MASKED]"

# Personal-data patterns with explicit character guards instead of \b, so a
# value written directly against Kanji or Kana is masked exactly as the same
# value written between ASCII spaces. Product identifiers and policy figures
# are deliberately NOT in this set: they must render intact for the answer to
# be usable, and they are not personal data.
_PERSONAL_DATA_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # Individual number, hyphenated 4-4-4. A date (2026-07-12) is 4-2-2 and a
    # policy limit (1,000,000) carries no hyphen, so neither shape can match.
    ("individual_number", re.compile(r"(?<![0-9-])\d{4}-\d{4}-\d{4}(?![0-9-])"), _MASK),
    # Individual number as a bare 12-digit run. Restricted to a run that follows
    # an explicit cue, and the cue is preserved, so an ordinary long policy
    # reference is not mangled by a length heuristic.
    (
        "individual_number",
        re.compile(
            r"((?:個人番号|マイナンバー|my\s*number)\s*[:：]?\s*)(?<![0-9])\d{12}(?![0-9])",
            re.IGNORECASE,
        ),
        r"\g<1>" + _MASK,
    ),
    # Domestic phone numbers, hyphenated.
    ("phone", re.compile(r"(?<![0-9-])0\d{1,4}-\d{1,4}-\d{4}(?![0-9-])"), _MASK),
    # E-mail addresses; the trailing guard stops the match absorbing an adjacent
    # Kanji particle.
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"), _MASK),
)

# Credential shapes the framework's own detector does not carry. Kept as an
# ADDITION to detect_credentials(), never as a replacement: a local set narrower
# than the framework's is a bypass, because the framework then raises inside the
# node wrapper and the node's own containment is discarded with the delta.
_EXTRA_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credential_assignment",
        re.compile(
            r"(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
)


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse *value* as a real, finite number inside ``[lo, hi]``; else ``None``.

    Rejects booleans (``isinstance(True, int)`` is true in Python), non-numeric
    types, and NaN / +-Infinity. Callers treat ``None`` as "not declared" and
    fall back to their own documented default, so an invalid declared value
    degrades to a known state instead of disabling the check it feeds.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


def resolve_repo_path(declared: Any) -> Optional[Path]:
    """Resolve a repository-relative declared path, or None if absent/escaping.

    A declared path is not caller data, but a configuration file can still be
    wrong: a value resolving outside the repository is refused rather than
    followed.
    """
    if not isinstance(declared, str) or not declared:
        return None
    candidate = (_REPO_ROOT / declared).resolve()
    if not candidate.is_relative_to(_REPO_ROOT):
        return None
    return candidate


def is_inert_token(value: Any) -> bool:
    """True when *value* is a short identifier over the closed inert alphabet."""
    return isinstance(value, str) and bool(_INERT_TOKEN_RE.match(value))


def safe_field_label(name: Any, position: int) -> str:
    """Return a name safe to echo in a refusal, or a positional label instead."""
    if isinstance(name, str) and _SAFE_FIELD_NAME_RE.match(name):
        return name
    return f"field #{position}"


def strip_markup(text: str) -> str:
    """Remove inline markup and invisible characters, joining what they split."""
    return _MARKUP_TAG_RE.sub("", _INVISIBLE_RE.sub("", text))


def screen_text(text: Any) -> Optional[str]:
    """Return the name of the first override pattern *text* carries, else None.

    Screens the value twice: as received, so a control token is caught before a
    strip could remove it, and with markup removed, so a directive spliced
    across inline tags is caught after the strip reassembles it.
    """
    if not isinstance(text, str) or not text:
        return None
    for candidate in (text, strip_markup(text)):
        for name, pattern in _CONTROL_TOKEN_PATTERNS:
            if pattern.search(candidate):
                return name
        for name, pattern in _DIRECTIVE_PATTERNS:
            if pattern.search(candidate):
                return name
    return None


def screen_structure(value: Any, _depth: int = 0) -> Optional[str]:
    """Screen every string leaf AND every mapping key, depth-first.

    Applied to the PARSED request rather than to the raw body, so a directive
    hidden behind JSON ``\\u`` escapes — absent from the raw text, present in the
    parsed value — is still caught. Keys are screened because a hostile field
    name reaches the same audit and refusal paths a value does. Depth is bounded
    so a deeply nested body cannot exhaust the stack.
    """
    if _depth > 8:
        return "nesting_depth_exceeded"
    if isinstance(value, str):
        return screen_text(value)
    if isinstance(value, dict):
        for key, nested in value.items():
            found = screen_text(key) if isinstance(key, str) else None
            if found:
                return found
            found = screen_structure(nested, _depth + 1)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = screen_structure(item, _depth + 1)
            if found:
                return found
    return None


def mask_personal_data(text: str) -> tuple[str, list[str]]:
    """Mask personal data in *text*; return the masked text and the kinds found.

    Only the kinds are returned — never the matched values — so a caller can
    audit that masking happened without the value re-entering a log line.
    """
    masked = text
    kinds: list[str] = []
    for kind, pattern, replacement in _PERSONAL_DATA_PATTERNS:
        masked, count = pattern.subn(replacement, masked)
        if count and kind not in kinds:
            kinds.append(kind)
    return masked, kinds


def detect_output_credentials(text: Any) -> Optional[str]:
    """Return the name of the first credential shape in *text*, else None.

    Delegates to the framework's own ``detect_credentials`` so the refusal set
    is exactly the block set the framework enforces one layer later. The extra
    patterns are additions the framework does not carry — an assignment line
    from a mis-pasted operations note is not a credential SHAPE, but it is
    exactly what such a note looks like.
    """
    if not isinstance(text, str) or not text:
        return None
    findings = detect_credentials(text)
    if findings:
        return str(findings[0]["type"])
    for name, pattern in _EXTRA_CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return name
    return None


class Service:
    """Domain service seam for INS-C2-031.

    The shipped pipeline answers offline over the product knowledge base bundled
    with ``RetrieveNode``, so this service is a documented no-op: ``fetch()``
    returns an empty result set rather than raising. It is the wiring seam for a
    real product-master or vector-store retriever, which slots in without
    changing the node contracts that call it.
    """

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch domain data for the given query.

        Returns ``{"results": []}``. The bundled knowledge base in
        ``RetrieveNode`` is the shipped source of truth; an external retriever
        wires in here.
        """
        return {"results": []}
