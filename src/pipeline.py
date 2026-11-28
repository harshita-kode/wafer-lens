"""One call that runs the whole analysis: classify -> similar wafers -> retrieve -> report."""
import logging
import time
from dataclasses import asdict, dataclass, field

from src.config import SIMILAR_K
from src.report.writer import write_report

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    prediction: dict
    patterns: list[str]
    report_markdown: str
    sources: list[dict]
    similar_wafers: list[dict]
    generated_by: str
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    report_body: str = ""
    wafer_facts: dict | None = None
    investigation: dict | None = None
    agent_steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class WaferAnalysisPipeline:
    mode = "pipeline"
    """Holds every component once; analyze() can then be called many times."""

    def __init__(self, classifier=None, retriever=None, similar_index=None, llm=None,
                 similar_k: int = SIMILAR_K, use_similar: bool = True):
        if classifier is None:
            from src.classifier.predict import get_classifier
            classifier = get_classifier()
        if retriever is None:
            from src.rag.retriever import get_retriever
            retriever = get_retriever()
        if similar_index is None and use_similar:
            similar_index = self._try_load_similar_index()
        if llm is None:
            llm = self._try_load_llm()

        self.classifier, self.retriever = classifier, retriever
        self.similar_index, self.llm, self.similar_k = similar_index, llm, similar_k

    @staticmethod
    def _try_load_similar_index():
        try:
            from src.similar.index import get_similar_index
            return get_similar_index()
        except Exception as e:
            logger.warning("Similar-wafer index unavailable: %s", e)
            return None

    @staticmethod
    def _try_load_llm():
        try:
            from src.llm.client import get_llm
            return get_llm()
        except Exception as e:
            logger.warning("LLM unavailable, reports will use the template: %s", e)
            return None

    def analyze(self, wafer_map) -> AnalysisResult:
        timings, warnings = {}, []

        start = time.perf_counter()
        prediction = self.classifier.predict(wafer_map)          # raises ValueError on bad input
        timings["classify"] = _ms(start)

        similar = []
        start = time.perf_counter()
        if self.similar_index is not None:
            similar = self.similar_index.find(self.classifier.embed(wafer_map), k=self.similar_k)
        else:
            warnings.append("Similar-wafer search unavailable")
        timings["similar"] = _ms(start)

        start = time.perf_counter()
        retrieval = self.retriever.retrieve_for_prediction(prediction)
        timings["retrieve"] = _ms(start)

        start = time.perf_counter()
        report = write_report(prediction, retrieval, llm=self.llm, similar=similar)
        timings["report"] = _ms(start)
        timings["total"] = round(sum(timings.values()), 1)

        return AnalysisResult(
            prediction=prediction,
            patterns=report.patterns,
            report_markdown=report.markdown,
            sources=report.sources,
            similar_wafers=similar,
            generated_by=report.generated_by,
            warnings=warnings + report.warnings,
            timings_ms=timings,
            report_body=report.body,
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


_default: WaferAnalysisPipeline | None = None


def get_pipeline() -> WaferAnalysisPipeline:
    global _default
    if _default is None:
        _default = WaferAnalysisPipeline()
    return _default


def analyze(wafer_map) -> AnalysisResult:
    return get_pipeline().analyze(wafer_map)