"""System prompt, citation relabelling and user-prompt construction for defect reports."""
import re

SYSTEM_PROMPT = """You are an assistant for semiconductor process engineers. You explain wafer-map defect patterns.

Rules:
- Use ONLY the information in the NOTES. Do not add causes, checks, tools or numbers that are not in the notes.
- After every claim taken from the notes, cite its source label exactly as written, for example [S1] or [S2][S3].
- Use cautious wording such as "often associated with", "commonly linked to" or "may indicate". Never say a pattern "is caused by" something.
- Do not state the confidence numbers; they are shown separately.
- If the notes do not cover something, say so briefly instead of guessing.
- Keep it short: 2-4 bullets per section, plain English, no extra headings.
- Start every bullet with "- " (a hyphen and a space). Do not use "*" or numbered lists."""


def source_key(source: dict) -> str:
    return source.get("url") or source.get("title")


def label_sources(chunks: list[dict]) -> dict:
    """Give every unique source a global label: {key: ("S1", source), ...}"""
    labels = {}
    for chunk in chunks:
        for s in chunk["sources"]:
            key = source_key(s)
            if key not in labels:
                labels[key] = (f"S{len(labels) + 1}", s)
    return labels


def renumber_citations(chunk: dict, labels: dict) -> str:
    """Rewrite a chunk's local [3] citations into global [S2]-style labels."""
    local_to_global = {s["id"]: labels[source_key(s)][0] for s in chunk["sources"]}

    def swap(match: re.Match) -> str:
        label = local_to_global.get(int(match.group(1)))
        return f"[{label}]" if label else ""

    return re.sub(r"\[(\d+)\]", swap, chunk["text"])


def format_source(s: dict) -> str:
    return f"{s['authors']} ({s['year']}). {s['title']}. {s['url']}"


def _task_instructions(prediction: dict, patterns: list[str]) -> str:
    if len(patterns) > 1:
        a, b = patterns[0], patterns[1]
        return (
            f"The classifier is UNCERTAIN between {a} and {b}. Write THREE sections, in this order:\n"
            f"**What this usually means:** possible causes for {a} and for {b}, naming the pattern in each bullet.\n"
            f"**Suggested checks:** checks for each pattern, most useful first.\n"
            f"**How to tell them apart:** 1-2 bullets on what distinguishes {a} from {b}, "
            f"using the 'Often confused with' notes if present."
        )
    if prediction["label"] == "none":
        return (
            "No systematic defect pattern was detected. Write ONE section:\n"
            "**What this usually means:** 1-2 bullets on what a wafer with no pattern looks like "
            "and what is normally monitored."
        )
    return (
        "Write TWO sections, in this order:\n"
        "**What this usually means:** the common causes of this pattern.\n"
        "**Suggested checks:** what engineers usually check, most useful first."
    )


def build_user_prompt(prediction: dict, patterns: list[str], notes: list[tuple[dict, str]],
                      labels: dict, feedback: list[str] | None = None) -> str:
    sources_block = "\n".join(f"[{label}] {format_source(s)}" for label, s in labels.values())
    notes_block = "\n\n".join(
        f"### Note {i}: {chunk['pattern']} | {chunk['section']}\n{text}"
        for i, (chunk, text) in enumerate(notes, start=1)
    )
    prompt = (
        f"PREDICTION\nPredicted pattern: {prediction['label']}\n"
        f"Patterns to explain: {', '.join(patterns)}\n\n"
        f"SOURCES\n{sources_block}\n\n"
        f"NOTES\n{notes_block}\n\n"
        f"TASK\n{_task_instructions(prediction, patterns)}"
    )
    if feedback:
        prompt += ("\n\nREVISION\nYour previous answer had these problems:\n"
                   + "\n".join(f"- {f}" for f in feedback)
                   + "\nWrite the whole answer again, fixing them.")
    return prompt