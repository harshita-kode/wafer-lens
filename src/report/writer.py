"""Turn a prediction + retrieved notes into a cited plain-English report."""
import re
from dataclasses import dataclass, field

from src.llm.client import LLMClient, get_llm
from src.report.prompts import (SYSTEM_PROMPT, build_user_prompt, format_source,
                                label_sources, renumber_citations)

CITATION = re.compile(r"\[(S\d+)\]")

BULLET = re.compile(r"^(\s*)[*•]\s+", flags=re.MULTILINE)


def normalize_bullets(text: str) -> str:
    """Turn '* item' or '• item' into '- item'. Bold headings like '**X:**' are untouched."""
    return BULLET.sub(r"\1- ", text)


@dataclass
class Report:
    markdown: str
    patterns: list[str]
    sources: list[dict]
    generated_by: str                 # "llm" or "template"
    body: str  
    warnings: list[str] = field(default_factory=list)


def format_header(prediction: dict) -> str:
    by = prediction.get("resolved_by")
    if by == "investigation":
        return (f"**Pattern detected:** {prediction['label']} (evidence score {prediction['confidence']:.0%}), "
                f"resolved by investigation; the CNN alone said {prediction['cnn_label']} "
                f"({prediction['cnn_confidence']:.0%})")
    if by == "unresolved":
        a, b = prediction["top_k"][:2]
        return (f"**Pattern detected:** uncertain between {a['label']} (evidence {a['confidence']:.0%}) "
                f"and {b['label']} ({b['confidence']:.0%}); the CNN alone said {prediction['cnn_label']} "
                f"({prediction['cnn_confidence']:.0%})")

    line = f"**Pattern detected:** {prediction['label']} ({prediction['confidence']:.0%} confidence)"
    if prediction["uncertain"] and len(prediction["top_k"]) > 1:
        second = prediction["top_k"][1]
        line += f", uncertain; possibly {second['label']} ({second['confidence']:.0%})"
    return line

def format_facts(facts: dict) -> str:
    zones = facts["failure_share_by_zone"]
    line = f"**This wafer:** {facts['failed_dies']} of {facts['dies']} dies failed ({facts['fail_rate']:.0%})"
    if facts["failed_dies"]:
        zone, share = max(zones.items(), key=lambda kv: kv[1])
        line += f"; {share:.0%} of failures are in the {zone} zone"
    return line + "."

def format_similar(similar: list[dict], predicted_label: str) -> str:
    same = sum(s["label"] == predicted_label for s in similar)
    lines = [f"**Similar past wafers:** {same} of the {len(similar)} most similar "
             f"past wafers were labelled {predicted_label}."]
    lines += [f"- {s['label']} (similarity {s['similarity']:.2f})" for s in similar]
    return "\n".join(lines)

def _bullets(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("- ")]


def template_body(patterns: list[str], notes: list[tuple[dict, str]]) -> str:
    """Fallback report built straight from the notes, no LLM."""
    multi = len(patterns) > 1
    lines = []
    for heading, section in [("What this usually means", "Common causes"),
                             ("Suggested checks", "What engineers usually check")]:
        lines += ["", f"**{heading}:**"]
        for chunk, text in notes:
            if chunk["section"] == section and chunk["pattern"] in patterns:
                for b in _bullets(text):
                    lines.append(f"- *{chunk['pattern']}:* {b[2:]}" if multi else b)
    return "\n".join(lines).strip()


def write_report(prediction: dict, retrieval: dict, llm: LLMClient | None = None,
                 similar: list[dict] | None = None, facts: dict | None = None,
                 feedback: list[str] | None = None, use_llm: bool = True) -> Report:
    chunks, patterns = retrieval["chunks"], retrieval["patterns"]
    labels = label_sources(chunks)
    notes = [(c, renumber_citations(c, labels)) for c in chunks]
    warnings = []

    # 1. Ask the LLM (unless told not to); fall back to a template if anything goes wrong
    if not use_llm:
        body, generated_by = template_body(patterns, notes), "template"
    else:
        try:
            llm = llm or get_llm()
            body = llm.generate(SYSTEM_PROMPT,
                                build_user_prompt(prediction, patterns, notes, labels, feedback=feedback))
            if not body:
                raise ValueError("empty response")
            generated_by = "llm"
        except Exception as e:
            warnings.append(f"LLM unavailable, used template report ({e})")
            body, generated_by = template_body(patterns, notes), "template"

    # 2. Check citations against the sources we actually provided
    known = {label for label, _ in labels.values()}
    cited = set(CITATION.findall(body))
    invented = cited - known
    if invented:
        warnings.append(f"Removed citations not in the notes: {sorted(invented)}")
        body = CITATION.sub(lambda m: m.group(0) if m.group(1) in known else "", body)
    valid_cited = cited & known
    if generated_by == "llm" and not valid_cited:
        warnings.append("The LLM response contains no citations")

    # 3. Assemble: header (code) + body (LLM) + sources (code)
    cited_sources = [{"label": label, **s} for label, s in labels.values() if label in valid_cited]
    parts = [format_header(prediction)]
    if facts:
        parts.append(format_facts(facts))
    parts.append(body)
    if similar:
        parts.append(format_similar(similar, prediction["label"]))

    return Report(
        markdown="\n\n".join(parts),
        patterns=patterns,
        sources=cited_sources,
        generated_by=generated_by,
        body=body,
        warnings=warnings,
    )


if __name__ == "__main__":
    import numpy as np

    from src.classifier.predict import get_classifier
    from src.config import DATA_DIR
    from src.rag.retriever import get_retriever

    data = np.load(DATA_DIR / "processed" / "wafers_64.npz")
    X, y = data["X_test"], data["y_test"]
    clf, retriever = get_classifier(), get_retriever()

    preds = clf.predict_batch(list(X[:300]))
    confident = next(i for i, p in enumerate(preds) if not p["uncertain"] and p["label"] != "none")
    uncertain = next(i for i, p in enumerate(preds) if p["uncertain"])

    for i in [confident, uncertain]:
        report = write_report(preds[i], retriever.retrieve_for_prediction(preds[i]))
        print("=" * 70)
        print(f"True label: {clf.classes[y[i]]}   |   generated by: {report.generated_by}")
        print("=" * 70)
        print(report.markdown)
        for w in report.warnings:
            print(f"\nWARNING: {w}")
        print()