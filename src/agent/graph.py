"""LangGraph agent: investigates uncertain wafers, skips unneeded LLM calls and checks its own report."""
import operator
import time
from typing import Annotated, Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from src.agent.checks import review_report
from src.agent.facts import wafer_facts
from src.agent.investigate import neighbour_shares, resolve, tta_probabilities
from src.config import AGENT_MAX_WRITES, SIMILAR_K
from src.pipeline import AnalysisResult, WaferAnalysisPipeline
from src.report.writer import write_report


class AgentState(TypedDict, total=False):
    wafer: Any
    prediction: dict
    facts: dict
    similar: list
    investigation: dict
    decision: dict
    retrieval: dict
    report: Any
    feedback: list
    writes: int
    steps: Annotated[list, operator.add]


class WaferAgent:
    mode = "agent"
    def __init__(self, classifier=None, retriever=None, similar_index=None, llm=None,
                 similar_k: int = SIMILAR_K, use_similar: bool = True, max_writes: int = AGENT_MAX_WRITES):
        base = WaferAnalysisPipeline(classifier, retriever, similar_index, llm, similar_k, use_similar)
        self.classifier, self.retriever = base.classifier, base.retriever
        self.similar_index, self.llm = base.similar_index, base.llm
        self.similar_k, self.max_writes = similar_k, max_writes
        self.graph = self._build()

    # ---------- graph ----------

    def _build(self):
        g = StateGraph(AgentState)
        for name, fn in [("classify", self._classify), ("measure", self._measure),
                         ("similar", self._similar), ("investigate", self._investigate),
                         ("decide", self._decide), ("retrieve", self._retrieve),
                         ("write", self._write), ("review", self._review), ("brief", self._brief)]:
            g.add_node(name, self._timed(name, fn))

        g.add_edge(START, "classify")
        g.add_edge("classify", "measure")
        g.add_edge("measure", "similar")
        g.add_conditional_edges("similar", self._after_similar,
                                {"investigate": "investigate", "decide": "decide"})
        for node in ("investigate", "decide"):
            g.add_conditional_edges(node, self._after_decision, {"brief": "brief", "retrieve": "retrieve"})
        g.add_edge("retrieve", "write")
        g.add_edge("write", "review")
        g.add_conditional_edges("review", self._after_review, {"write": "write", "done": END})
        g.add_edge("brief", END)
        return g.compile()

    @staticmethod
    def _timed(name, fn):
        def node(state):
            start = time.perf_counter()
            updates, summary = fn(state)
            updates["steps"] = [{"node": name, "summary": summary,
                                 "ms": round((time.perf_counter() - start) * 1000, 1)}]
            return updates
        return node

    # ---------- nodes ----------

    def _classify(self, s):
        p = self.classifier.predict(s["wafer"])
        flag = " (uncertain)" if p["uncertain"] else ""
        return {"prediction": p}, f"CNN predicts {p['label']} at {p['confidence']:.0%}{flag}"

    def _measure(self, s):
        f = wafer_facts(s["wafer"])
        summary = f"{f['fail_rate']:.0%} of dies failed"
        if f["failed_dies"]:
            zone, share = max(f["failure_share_by_zone"].items(), key=lambda kv: kv[1])
            summary += f", {share:.0%} of them in the {zone} zone"
        return {"facts": f}, summary

    def _similar(self, s):
        if self.similar_index is None:
            return {"similar": []}, "Similar-wafer search unavailable"
        similar = self.similar_index.find(self.classifier.embed(s["wafer"]), k=self.similar_k)
        label = s["prediction"]["label"]
        same = sum(w["label"] == label for w in similar)
        return {"similar": similar}, f"{same} of {len(similar)} similar past wafers are {label}"

    def _investigate(self, s):
        p = s["prediction"]
        r = resolve(p, tta_probabilities(self.classifier, s["wafer"]),
                    neighbour_shares(s["similar"], self.classifier.classes), self.classifier.classes)
        r.update(cnn_label=p["label"], cnn_confidence=p["confidence"])
        a, b = r["candidates"]
        decision = {"label": r["label"], "confidence": r["scores"][r["label"]], "uncertain": r["uncertain"],
                    "top_k": [{"label": a, "confidence": r["scores"][a]},
                              {"label": b, "confidence": r["scores"][b]}],
                    "resolved_by": r["resolved_by"], "cnn_label": p["label"], "cnn_confidence": p["confidence"]}
        if r["uncertain"]:
            summary = f"Still ambiguous: {a} {r['scores'][a]:.0%} vs {b} {r['scores'][b]:.0%}"
        else:
            summary = f"Resolved to {a}: evidence {r['scores'][a]:.0%} vs {r['scores'][b]:.0%} for {b}"
        return {"investigation": r, "decision": decision}, summary

    def _decide(self, s):
        p = s["prediction"]
        decision = {**p, "resolved_by": "cnn", "cnn_label": p["label"], "cnn_confidence": p["confidence"]}
        return {"decision": decision}, "Confident prediction; no investigation needed"

    def _retrieve(self, s):
        retrieval = self.retriever.retrieve_for_prediction(s["decision"])
        return {"retrieval": retrieval}, (f"Retrieved {len(retrieval['chunks'])} note sections "
                                          f"for {', '.join(retrieval['patterns'])}")

    def _write(self, s):
        report = write_report(s["decision"], s["retrieval"], llm=self.llm, similar=s.get("similar"),
                              facts=s.get("facts"), feedback=s.get("feedback") or None)
        verb = "Rewrote the report using review feedback" if s.get("feedback") else "Wrote the report"
        return {"report": report, "writes": s.get("writes", 0) + 1}, f"{verb} ({report.generated_by})"

    def _review(self, s):
        report = s["report"]
        if report.generated_by != "llm":
            return {"feedback": []}, "Template report; review skipped"
        issues = review_report(report, s["decision"])
        summary = "Report passed all checks" if not issues else f"{len(issues)} issue(s): " + " ".join(issues)
        return {"feedback": issues}, summary

    def _brief(self, s):
        retrieval = self.retriever.retrieve_for_prediction(s["decision"])
        report = write_report(s["decision"], retrieval, similar=s.get("similar"),
                              facts=s.get("facts"), use_llm=False)
        return {"retrieval": retrieval, "report": report}, "No defect pattern: short report from notes, LLM skipped"

    # ---------- routing ----------

    @staticmethod
    def _after_similar(s):
        return "investigate" if s["prediction"]["uncertain"] else "decide"

    @staticmethod
    def _after_decision(s):
        d = s["decision"]
        return "brief" if d["label"] == "none" and not d["uncertain"] else "retrieve"

    def _after_review(self, s):
        return "write" if s.get("feedback") and s.get("writes", 0) < self.max_writes else "done"

    # ---------- public API ----------

    def analyze(self, wafer_map) -> AnalysisResult:
        final = self.graph.invoke({"wafer": np.asarray(wafer_map), "writes": 0, "steps": []})
        report, steps = final["report"], final["steps"]

        warnings = list(report.warnings)
        if self.similar_index is None:
            warnings.append("Similar-wafer search unavailable")
        if final.get("feedback"):
            warnings.append("Report still had issues after revision: " + " ".join(final["feedback"]))

        timings = {}
        for step in steps:
            timings[step["node"]] = round(timings.get(step["node"], 0) + step["ms"], 1)
        timings["total"] = round(sum(step["ms"] for step in steps), 1)

        return AnalysisResult(
            prediction=final["decision"], patterns=report.patterns, report_markdown=report.markdown,
            sources=report.sources, similar_wafers=final.get("similar", []),
            generated_by=report.generated_by, warnings=warnings, timings_ms=timings,
            report_body=report.body, wafer_facts=final.get("facts"),
            investigation=final.get("investigation"), agent_steps=steps,
        )


_default: WaferAgent | None = None


def get_agent() -> WaferAgent:
    global _default
    if _default is None:
        _default = WaferAgent()
    return _default