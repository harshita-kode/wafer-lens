"""Code checks the agent runs on its own report before accepting it."""
import re

OVERCONFIDENT = ("is caused by", "are caused by", "definitely", "certainly")


def expected_sections(patterns: list[str], label: str) -> list[str]:
    if len(patterns) > 1:
        return ["What this usually means", "Suggested checks", "How to tell them apart"]
    if label == "none":
        return ["What this usually means"]
    return ["What this usually means", "Suggested checks"]


def review_report(report, decision: dict) -> list[str]:
    """Problems found in the report, phrased as instructions the LLM can act on."""
    body, issues = report.body, []

    for section in expected_sections(report.patterns, decision["label"]):
        if f"**{section}:**" not in body:
            issues.append(f'Add the missing "{section}" section.')

    bullets = [line for line in body.splitlines() if line.strip().startswith("- ")]
    uncited = [b for b in bullets if not re.search(r"\[S\d+\]", b)]
    if uncited:
        issues.append(f"{len(uncited)} bullet(s) have no citation; cite every bullet with the [S#] labels provided.")

    if any(w.startswith("Removed citations") for w in report.warnings):
        issues.append("Use only the [S#] labels listed under SOURCES.")

    if any(phrase in body.lower() for phrase in OVERCONFIDENT):
        issues.append('Avoid definite wording such as "is caused by"; use cautious wording.')

    return issues