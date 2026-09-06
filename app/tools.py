"""Two deterministic tools: a fine calculator and a compliance timeline.

Neither one retrieves. The model never does the arithmetic and never resolves a
date; it gets a figure and a legal basis and reports them. Both read JSON fact
files where every tier and date carries the quote it came from, checked against
the regulation text in tests/test_facts.py.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings

Entity = Literal["undertaking", "sme", "smc", "eu_institution", "non_undertaking"]

ENTITY_LABELS: dict[str, str] = {
    "undertaking": "an undertaking",
    "sme": "an SME or start-up",
    "smc": "a small mid-cap enterprise",
    "eu_institution": "a Union institution, body, office or agency",
    "non_undertaking": "a person who is not an undertaking",
}


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    value: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    citation: str = ""
    summary: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class PenaltyArgs(BaseModel):
    tier_id: str
    entity: Entity = "undertaking"
    turnover_eur: float | None = Field(default=None, ge=0)


@lru_cache(maxsize=1)
def _penalties(path: str) -> dict[str, Any]:
    return json.loads(open(path, encoding="utf-8").read())


def load_penalties(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return _penalties(str(settings.data_dir / "penalties.json"))


def all_tiers(settings: Settings | None = None) -> dict[str, dict[str, Any]]:
    tiers: dict[str, dict[str, Any]] = {}
    for regime in load_penalties(settings)["regimes"]:
        for tier in regime["tiers"]:
            tiers[tier["id"]] = {**tier, "regime": regime}
    return tiers


def get_tier(tier_id: str, settings: Settings | None = None) -> dict[str, Any]:
    tiers = all_tiers(settings)
    if tier_id not in tiers:
        raise KeyError(f"unknown tier {tier_id!r}; known: {sorted(tiers)}")
    return tiers[tier_id]


def _modifier_for(tier_id: str, entity: str, regime: dict[str, Any]) -> dict[str, Any] | None:
    if entity not in ("sme", "smc"):
        return None
    for modifier in regime.get("modifiers", []):
        if modifier["id"] == entity and tier_id in modifier["applies_to_tiers"]:
            return modifier
    return None


def compute_penalty(
    tier_id: str,
    entity: Entity = "undertaking",
    turnover_eur: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Maximum administrative fine for one tier.

    The comparator does the work here. Art. 99 normally takes whichever of the
    fixed amount and the turnover percentage is higher. Art. 99(6) flips that to
    lower for SMEs on all three tiers. Art. 99(6a), added by the Digital
    Omnibus, flips it for SMCs on paragraphs 4 and 5 only, so an SMC that
    commits a prohibited practice gets no relief at all. That last bit is the
    part people state backwards.
    """
    tier = get_tier(tier_id, settings)
    regime = tier["regime"]
    fixed = float(tier["amount_eur"])
    caveats: list[str] = []

    if tier["comparator"] == "flat" or tier["turnover_pct"] is None:
        return {
            "ceiling_eur": fixed,
            "limb": "fixed",
            "comparator": "flat",
            "tier_id": tier_id,
            "legal_basis": f"{regime['regulation']} Art. {tier['paragraph']}",
            "modifier": None,
            "caveats": caveats,
            "explanation": f"{regime['regulation']} Art. {tier['paragraph']} sets a fixed ceiling.",
        }

    if entity == "non_undertaking":
        caveats.append(
            "The turnover limb applies to undertakings. For a person who is not "
            "an undertaking, only the fixed amount is available."
        )
        turnover_eur = None

    modifier = _modifier_for(tier_id, entity, regime)
    comparator = modifier["comparator"] if modifier else tier["comparator"]

    if entity == "smc" and modifier is None and regime["id"] == "ai_act_art99":
        caveats.append(
            f"The SMC cap in Art. 99(6a) covers paragraphs 4 and 5 only, so it "
            f"does not apply to {tier['paragraph']}. The higher-of comparison stands."
        )

    if turnover_eur is None:
        caveats.append(
            f"Turnover was not supplied, so the {tier['turnover_pct']:g}% alternative could "
            f"not be evaluated. The figure below is the fixed ceiling only."
        )
        ceiling, limb = fixed, "fixed"
    else:
        turnover_limb = turnover_eur * float(tier["turnover_pct"]) / 100.0
        if comparator == "lower":
            ceiling, limb = (
                (fixed, "fixed") if fixed <= turnover_limb else (turnover_limb, "turnover")
            )
        else:
            ceiling, limb = (
                (fixed, "fixed") if fixed >= turnover_limb else (turnover_limb, "turnover")
            )

    explanation = (
        f"{regime['regulation']} Art. {tier['paragraph']}: EUR {fixed:,.0f} or "
        f"{tier['turnover_pct']:g}% of total worldwide annual turnover, whichever is "
        f"{comparator}"
    )
    if modifier:
        explanation += f", as modified for {ENTITY_LABELS[entity]} by Art. {modifier['paragraph']}"

    return {
        "ceiling_eur": round(ceiling, 2),
        "limb": limb,
        "comparator": comparator,
        "tier_id": tier_id,
        "legal_basis": f"{regime['regulation']} Art. {tier['paragraph']}",
        "modifier": modifier["paragraph"] if modifier else None,
        "caveats": caveats,
        "explanation": explanation + ".",
    }


class DeadlineArgs(BaseModel):
    query: str
    on: date | None = None


@lru_cache(maxsize=1)
def _deadlines(path: str) -> list[dict[str, Any]]:
    payload = json.loads(open(path, encoding="utf-8").read())
    return sorted(payload["deadlines"], key=lambda item: item["date"])


def all_deadlines(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    return _deadlines(str(settings.data_dir / "deadlines.json"))


_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are",
     "when", "do", "does"}
)


def search_deadlines(query: str, settings: Settings | None = None) -> dict[str, Any] | None:
    """Best deadline by keyword overlap. Skips superseded entries."""
    terms = {word for word in query.lower().replace(".", " ").split() if word not in _STOPWORDS}
    best: dict[str, Any] | None = None
    best_score = 0
    for entry in all_deadlines(settings):
        if entry["status"] == "superseded":
            continue
        haystack = f"{entry['subject']} {entry['legal_basis']} {entry['reasoning']}".lower()
        score = sum(1 for term in terms if term in haystack)
        if score > best_score:
            best, best_score = entry, score
    return best


def deadline_predecessor(
    deadline_id: str, settings: Settings | None = None
) -> dict[str, Any] | None:
    for entry in all_deadlines(settings):
        if entry.get("superseded_by") == deadline_id:
            return entry
    return None


def lookup_deadline(
    query: str, on: date | None = None, settings: Settings | None = None
) -> dict[str, Any]:
    entry = search_deadlines(query, settings)
    if entry is None:
        raise LookupError(f"no compliance date matches {query!r}")

    applies = date.fromisoformat(entry["date"])
    today = on or date.today()
    previous = deadline_predecessor(entry["id"], settings)

    description = f"{entry['subject']} Applies from {applies.isoformat()} ({entry['status']})."
    if previous:
        description += (
            f" This date was moved from {previous['date']} by {entry.get('amended_by')}."
        )

    return {
        "deadline_id": entry["id"],
        "regulation": entry["regulation"],
        "applies_on": applies.isoformat(),
        "status": entry["status"],
        "subject": entry["subject"],
        "legal_basis": entry["legal_basis"],
        "days_remaining": (applies - today).days,
        "previously": previous["date"] if previous else None,
        "amended_by": entry.get("amended_by"),
        "description": description,
    }


TOOLS = {
    "compute_penalty": (
        "Compute the maximum administrative fine for an AI Act or GDPR "
        "infringement tier, applying the SME and SMC comparator rules."
    ),
    "lookup_deadline": (
        "Look up when an AI Act or GDPR obligation starts to apply, including "
        "dates moved by the Digital Omnibus."
    ),
}


def dispatch(name: str, args: dict[str, Any], settings: Settings | None = None) -> ToolResult:
    """Run a tool. Never raises; failures come back as a readable result.

    A malformed argument should cost the tool's contribution and nothing else.
    Retrieval can still carry a decent answer on its own.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    def elapsed() -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    if name not in TOOLS:
        return ToolResult(tool=name, ok=False, error=f"unknown tool {name!r}", args=args)

    try:
        if name == "compute_penalty":
            parsed = PenaltyArgs.model_validate(args)
            value = compute_penalty(
                parsed.tier_id, parsed.entity, parsed.turnover_eur, settings
            )
            summary = f"Maximum fine EUR {value['ceiling_eur']:,.0f} ({value['legal_basis']})."
        else:
            parsed_dl = DeadlineArgs.model_validate(args)
            value = lookup_deadline(parsed_dl.query, parsed_dl.on, settings)
            summary = value["description"]
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return ToolResult(
            tool=name, ok=False, error=f"invalid arguments: {problems}", args=args,
            latency_ms=elapsed(),
        )
    except (KeyError, LookupError, ValueError) as exc:
        return ToolResult(
            tool=name, ok=False, error=f"{name} produced no result: {exc}", args=args,
            latency_ms=elapsed(),
        )

    return ToolResult(
        tool=name,
        ok=True,
        value=value,
        citation=value.get("legal_basis", ""),
        summary=summary,
        args=args,
        latency_ms=elapsed(),
    )
