from types import SimpleNamespace

from src.agent.checks import review_report

CONFIDENT = {"label": "Scratch"}


def report(body, patterns=("Scratch",), warnings=()):
    return SimpleNamespace(body=body, patterns=list(patterns), warnings=list(warnings))


GOOD = "**What this usually means:**\n- Handling damage [S1]\n\n**Suggested checks:**\n- Handler paths [S2]"


def test_good_report_passes():
    assert review_report(report(GOOD), CONFIDENT) == []


def test_missing_section_is_flagged():
    issues = review_report(report("**What this usually means:**\n- Damage [S1]"), CONFIDENT)
    assert any("Suggested checks" in i for i in issues)


def test_uncited_bullet_is_flagged():
    issues = review_report(report(GOOD.replace(" [S2]", "")), CONFIDENT)
    assert any("1 bullet(s) have no citation" in i for i in issues)


def test_uncertain_needs_comparison_section():
    issues = review_report(report(GOOD, patterns=("Loc", "Edge-Loc")), {"label": "Loc"})
    assert any("How to tell them apart" in i for i in issues)


def test_overconfident_wording_is_flagged():
    issues = review_report(report(GOOD.replace("Handling damage", "It is caused by handling")), CONFIDENT)
    assert any("cautious" in i for i in issues)