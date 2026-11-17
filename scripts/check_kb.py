"""Check every knowledge-base note follows the template."""
import json
import re
import sys
from pathlib import Path

import frontmatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "knowledge_base"
CLASSES = json.load(open(PROJECT_ROOT / "data" / "label_map.json"))

PATTERN_HEADINGS = ["What it looks like", "Common causes",
                    "What engineers usually check", "Often confused with"]
GENERAL_HEADINGS = ["What a wafer map is", "How wafers are tested",
                    "What yield means", "Why spatial patterns matter"]


def check_note(path: Path) -> list[str]:
    errors = []
    note = frontmatter.load(path)
    meta, body = note.metadata, note.content

    pattern = meta.get("pattern")
    if pattern != "general" and pattern not in CLASSES:
        errors.append(f"pattern '{pattern}' is not 'general' or one of {CLASSES}")

    if not meta.get("title"):
        errors.append("missing title")

    sources = meta.get("sources") or []
    if not sources:
        errors.append("no sources")
    source_ids = set()
    for s in sources:
        for field in ["id", "title", "authors", "year", "url"]:
            if not s.get(field):
                errors.append(f"source {s.get('id', '?')} missing '{field}'")
        source_ids.add(s.get("id"))

    headings = re.findall(r"^## (.+)$", body, flags=re.MULTILINE)
    expected = GENERAL_HEADINGS if pattern == "general" else PATTERN_HEADINGS
    for h in expected:
        if h not in headings:
            errors.append(f"missing heading '## {h}'")

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    for n in cited - source_ids:
        errors.append(f"cites [{n}] but there is no source with id {n}")

    words = len(body.split())
    if words > 500:
        errors.append(f"{words} words (aim for under 400)")

    return errors


def main() -> int:
    notes = sorted(KB_DIR.glob("*.md"))
    found = {frontmatter.load(p).metadata.get("pattern") for p in notes}

    total_errors = 0
    for path in notes:
        errors = check_note(path)
        status = "OK " if not errors else "FIX"
        print(f"[{status}] {path.name}")
        for e in errors:
            print(f"        - {e}")
        total_errors += len(errors)

    missing = [c for c in CLASSES + ["general"] if c not in found]
    if missing:
        print(f"\nMissing notes for: {missing}")
        total_errors += len(missing)

    print(f"\n{len(notes)} notes checked, {total_errors} problem(s)")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())