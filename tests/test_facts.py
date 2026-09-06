"""Checks the JSON fact files against the regulation text they quote.

The tools are only as good as penalties.json and deadlines.json. A typo in a
ceiling gives you a precise, confident, wrong answer, and no amount of retrieval
quality catches that. So every quoted fact gets located in its source document
here. Anything derived instead of quoted has to carry its reasoning.
"""

from app.corpus import normalize
from app.tools import all_deadlines, load_penalties


def test_every_tier_quote_appears_in_its_regulation(settings, source_text):
    for regime in load_penalties(settings)["regimes"]:
        for tier in regime["tiers"]:
            quote = normalize(tier["quote"])
            assert quote, f"{tier['id']} has no quote"
            assert quote in source_text[regime["source_key"]], (
                f"{tier['id']} quote not found in {regime['source_key']}"
            )


def test_every_modifier_quote_appears_in_its_regulation(settings, source_text):
    for regime in load_penalties(settings)["regimes"]:
        for modifier in regime.get("modifiers", []):
            quote = normalize(modifier["quote"])
            assert quote in source_text[regime["source_key"]], (
                f"{modifier['id']} quote not found in {regime['source_key']}"
            )


def test_every_tier_amount_appears_in_its_quote(settings):
    """The number the calculator returns must be in the text it cites."""
    for regime in load_penalties(settings)["regimes"]:
        for tier in regime["tiers"]:
            spaced = f"{tier['amount_eur']:,}".replace(",", " ")
            assert spaced in normalize(tier["quote"]), f"{tier['id']} amount not in its quote"


def test_verbatim_deadline_quotes_appear_in_their_source(settings, source_text):
    for entry in all_deadlines(settings):
        if not entry["verbatim"]:
            continue
        assert normalize(entry["quote"]) in source_text[entry["source_key"]], (
            f"{entry['id']} claims to be verbatim but is not in {entry['source_key']}"
        )


def test_derived_deadlines_carry_their_reasoning(settings):
    for entry in all_deadlines(settings):
        if entry["verbatim"]:
            continue
        assert len(entry["reasoning"]) > 40, f"{entry['id']} is derived but barely explained"


def test_the_omnibus_deferral_is_recorded_in_both_directions(settings):
    entries = {entry["id"]: entry for entry in all_deadlines(settings)}
    current = entries["aia_highrisk_annex_iii"]
    original = entries["aia_highrisk_annex_iii_original"]
    assert current["date"] == "2027-12-02"
    assert original["date"] == "2026-08-02"
    assert original["status"] == "superseded"
    assert original["superseded_by"] == current["id"]


def test_smc_modifier_deliberately_excludes_the_prohibited_practices_tier(settings):
    """Pin the asymmetry. Widening it later should fail loudly, not quietly."""
    regime = next(r for r in load_penalties(settings)["regimes"] if r["id"] == "ai_act_art99")
    smc = next(m for m in regime["modifiers"] if m["id"] == "smc")
    sme = next(m for m in regime["modifiers"] if m["id"] == "sme")
    assert "aia_prohibited" not in smc["applies_to_tiers"]
    assert "aia_prohibited" in sme["applies_to_tiers"]


def test_the_superseded_source_is_flagged_as_not_current(provisions):
    originals = [p for p in provisions if p.source_key == "ai_act_original"]
    assert originals and not any(p.is_current_text for p in originals)
