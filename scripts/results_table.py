"""Print every evaluation result as Markdown tables, ready to paste into the README."""
import json

from src.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"


def load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def classifier_table(m: dict) -> str:
    lines = [f"**Test accuracy {m['test_accuracy']:.1%} · macro F1 {m['test_macro_f1']:.3f}** "
             f"(best epoch {m['model_epoch']})\n",
             "| Pattern | Precision | Recall | F1 | Test wafers |",
             "|---|---|---|---|---|"]
    for name, s in m["per_class"].items():
        lines.append(f"| {name} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {s['support']} |")
    return "\n".join(lines)


def main() -> None:
    blocks = []

    if m := load("test_metrics.json"):
        blocks.append("### Defect classifier (CNN)\n\n" + classifier_table(m))

    if s := load("similar_eval.json"):
        blocks.append("### Similar past wafers\n\n"
                      "| Metric | Value |\n|---|---|\n"
                      f"| precision@{s['k']} (neighbours sharing the true label) | {s['precision_at_k']:.1%} |\n"
                      f"| Majority vote of {s['k']} neighbours, accuracy | {s['knn_vote_accuracy']:.1%} |\n"
                      f"| Wafers evaluated | {s['wafers']} |")

    if r := load("retriever_eval.json"):
        blocks.append("### Knowledge retrieval\n\n"
                      "| Metric | Value |\n|---|---|\n"
                      f"| Pattern-filtered retrieval checks passed | {r['pattern_checks_passed']}/{r['pattern_checks_total']} |\n"
                      f"| Free-text search, top-1 hit rate | {r['top1_hit_rate']:.0%} |\n"
                      f"| Free-text search, top-3 hit rate | {r['top3_hit_rate']:.0%} |")

    if e := load("report_eval.json"):
        s = e["summary"]
        rows = [("Reports evaluated", s["reports"]),
                ("Written by the LLM (not the fallback)", f"{s['llm_used_rate']:.0%}"),
                ("All requested sections present", f"{s.get('sections_ok_rate', 0):.0%}"),
                ("Bullets with a citation", f"{s['avg_bullet_citation_rate']:.0%}"),
                ("Reports with overconfident wording", s["reports_with_overconfident_wording"]),
                ("Uncertain reports covering both patterns",
                 f"{s['uncertain_mention_both_rate']:.0%}" if s.get("uncertain_mention_both_rate") is not None else "n/a")]
        if s.get("avg_seconds_uncached") is not None:
            rows.append(("Report time, fresh LLM call", f"{s['avg_seconds_uncached']:.1f} s"))
        if s.get("avg_seconds_cached") is not None:
            rows.append(("Report time, cached", f"{s['avg_seconds_cached'] * 1000:.0f} ms"))
        blocks.append("### LLM reports\n\n| Metric | Value |\n|---|---|\n" +
                      "\n".join(f"| {k} | {v} |" for k, v in rows))

    if a := load("agent_eval.json"):
        blocks.append("### Investigation agent (uncertain wafers)\n\n"
                      "| Metric | Value |\n|---|---|\n"
                      f"| Uncertain test wafers | {a['uncertain_wafers']} |\n"
                      f"| Top-1 accuracy, CNN alone | {a['cnn_top1_accuracy']:.1%} |\n"
                      f"| Top-1 accuracy, with agent | {a['agent_top1_accuracy']:.1%} |\n"
                      f"| Resolved by investigation | {a['resolved_rate']:.0%} |\n"
                      f"| Accuracy when resolved | {a['resolved_accuracy']:.1%} |\n"
                      f"| True pattern among the two reported (unresolved) | {a['unresolved_top2_coverage']:.1%} |")

    print("\n\n".join(blocks) if blocks else "No results found in results/")


if __name__ == "__main__":
    main()