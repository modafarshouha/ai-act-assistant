import pytest

from app.tools import all_tiers, compute_penalty, dispatch, lookup_deadline

BIG = 800_000_000


def test_every_tier_is_reachable(settings):
    tiers = all_tiers(settings)
    assert set(tiers) == {
        "aia_prohibited",
        "aia_obligations",
        "aia_misinformation",
        "aia_eu_body_prohibited",
        "aia_eu_body_other",
        "gdpr_lower",
        "gdpr_upper",
    }


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("aia_prohibited", 35_000_000),
        ("aia_obligations", 15_000_000),
        ("aia_misinformation", 7_500_000),
    ],
)
def test_fixed_ceilings_match_article_99(tier, expected):
    assert compute_penalty(tier)["ceiling_eur"] == expected


def test_turnover_limb_wins_when_higher():
    result = compute_penalty("aia_prohibited", "undertaking", BIG)
    assert result["ceiling_eur"] == 56_000_000
    assert result["limb"] == "turnover"


def test_sme_takes_the_lower_limb_on_every_tier():
    for tier in ("aia_prohibited", "aia_obligations", "aia_misinformation"):
        result = compute_penalty(tier, "sme", BIG)
        assert result["comparator"] == "lower"
        assert result["ceiling_eur"] == compute_penalty(tier)["ceiling_eur"]


def test_smc_relief_applies_to_paragraphs_4_and_5():
    for tier in ("aia_obligations", "aia_misinformation"):
        assert compute_penalty(tier, "smc", BIG)["comparator"] == "lower"


def test_smc_gets_no_relief_on_a_prohibited_practice():
    """Art. 99(6a) names paragraphs 4 and 5 only, so 99(3) keeps higher-of.

    Most commonly mis-stated part of the regime. An SME on identical facts pays
    EUR 21 million less.
    """
    smc = compute_penalty("aia_prohibited", "smc", BIG)
    sme = compute_penalty("aia_prohibited", "sme", BIG)
    assert smc["ceiling_eur"] == 56_000_000
    assert sme["ceiling_eur"] == 35_000_000
    assert smc["comparator"] == "higher"
    assert smc["modifier"] is None
    assert any("99(6a)" in caveat for caveat in smc["caveats"])


def test_article_100_is_flat_and_ignores_turnover():
    result = compute_penalty("aia_eu_body_prohibited", "eu_institution", BIG)
    assert result["ceiling_eur"] == 1_500_000
    assert result["comparator"] == "flat"


def test_gdpr_tiers():
    assert compute_penalty("gdpr_lower")["ceiling_eur"] == 10_000_000
    assert compute_penalty("gdpr_upper", "undertaking", 1_000_000_000)["ceiling_eur"] == 40_000_000


def test_unknown_turnover_says_so_rather_than_guessing():
    result = compute_penalty("aia_prohibited")
    assert result["limb"] == "fixed"
    assert any("not supplied" in caveat for caveat in result["caveats"])


def test_non_undertaking_has_no_turnover_limb():
    result = compute_penalty("aia_prohibited", "non_undertaking", BIG)
    assert result["ceiling_eur"] == 35_000_000


def test_annex_iii_deadline_was_deferred_by_the_omnibus():
    result = lookup_deadline("Annex III high-risk obligations")
    assert result["applies_on"] == "2027-12-02"
    assert result["previously"] == "2026-08-02"
    assert "2026/1744" in result["amended_by"]


def test_deadline_search_skips_superseded_entries():
    assert lookup_deadline("Annex I high-risk obligations")["status"] != "superseded"


def test_dispatch_returns_a_result_for_an_unknown_tool():
    result = dispatch("frobnicate", {})
    assert not result.ok and "unknown tool" in result.error


def test_dispatch_reports_invalid_arguments_without_raising():
    result = dispatch("compute_penalty", {"tier_id": "aia_prohibited", "entity": "martian"})
    assert not result.ok and "entity" in result.error


def test_dispatch_reports_an_unknown_tier_without_raising():
    result = dispatch("compute_penalty", {"tier_id": "nope"})
    assert not result.ok and "nope" in result.error


def test_dispatch_carries_the_legal_basis_on_success():
    result = dispatch("compute_penalty", {"tier_id": "aia_prohibited", "entity": "smc",
                                          "turnover_eur": BIG})
    assert result.ok
    assert result.citation == "AI Act Art. 99(3)"
    assert "56,000,000" in result.summary
