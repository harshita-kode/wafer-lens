"""WaferLens HTTP API."""
import io
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import API_VERSION, FRONTEND_DIR, SAMPLES_DIR

from src.api.schemas import (AnalysisResponse, AnalyzeRequest, ErrorResponse,
                             HealthResponse, SampleList, SampleWafer, PastWafer)
from src.config import API_VERSION, SAMPLES_DIR

logger = logging.getLogger("waferlens.api")

MAX_SIDE = 512
MAX_UPLOAD_BYTES = 2_000_000
ERRORS = {400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
          422: {"model": ErrorResponse}}


def default_pipeline_factory():
    from src.pipeline import WaferAnalysisPipeline
    return WaferAnalysisPipeline()

def default_past_store_factory():
    from src.similar.store import PastWaferStore
    return PastWaferStore()

def check_size(wafer: np.ndarray) -> np.ndarray:
    if wafer.ndim == 2 and max(wafer.shape) > MAX_SIDE:
        raise ValueError(f"Wafer map is {wafer.shape}; the maximum is {MAX_SIDE}x{MAX_SIDE}")
    return wafer


def create_app(pipeline_factory=default_pipeline_factory, samples_dir=SAMPLES_DIR,
               past_store_factory=default_past_store_factory) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading WaferLens pipeline...")
        start = time.perf_counter()
        app.state.pipeline = pipeline_factory()
        app.state.samples_dir = samples_dir
        try:
            app.state.past_store = past_store_factory()
        except Exception as e:
            logger.warning("Past wafer maps unavailable: %s", e)
            app.state.past_store = None
        logger.info("Pipeline ready in %.1fs", time.perf_counter() - start)
        yield
        logger.info("Shutting down")

    app = FastAPI(
        title="WaferLens API",
        version=API_VERSION,
        description="Classifies wafer-map defect patterns and returns a cited root-cause report.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def get_pipeline(request: Request):
        return request.app.state.pipeline

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(pipeline=Depends(get_pipeline)):
        parts = {
            "classifier": pipeline.classifier is not None,
            "retriever": pipeline.retriever is not None,
            "similar_index": pipeline.similar_index is not None,
            "llm": pipeline.llm is not None,
        }
        return HealthResponse(
            status="ok" if all(parts.values()) else "degraded",
            model_epoch=getattr(pipeline.classifier, "model_epoch", None),
            version=API_VERSION,
            **parts,
        )

    @app.post("/analyze", response_model=AnalysisResponse, responses=ERRORS, tags=["analysis"])
    def analyze(body: AnalyzeRequest, pipeline=Depends(get_pipeline)):
        wafer = check_size(np.asarray(body.wafer_map))
        return pipeline.analyze(wafer).to_dict()

    @app.post("/analyze/file", response_model=AnalysisResponse, responses=ERRORS, tags=["analysis"])
    async def analyze_file(file: UploadFile = File(..., description="A 2D .npy wafer map"),
                           pipeline=Depends(get_pipeline)):
        if not (file.filename or "").lower().endswith(".npy"):
            raise HTTPException(status_code=400, detail="Upload a .npy file")

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 2 MB)")

        try:
            wafer = np.load(io.BytesIO(content), allow_pickle=False)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read the file as a numeric .npy array")

        result = await run_in_threadpool(pipeline.analyze, check_size(wafer))
        return result.to_dict()

    def sample_names(request: Request) -> list[str]:
        return sorted(p.stem for p in request.app.state.samples_dir.glob("*.npy"))

    @app.get("/samples", response_model=SampleList, tags=["samples"])
    def list_samples(request: Request):
        return SampleList(samples=sample_names(request))

    @app.get("/samples/{name}", response_model=SampleWafer,
             responses={404: {"model": ErrorResponse}}, tags=["samples"])
    def get_sample(name: str, request: Request):
        if name not in sample_names(request):
            raise HTTPException(status_code=404, detail=f"Unknown sample '{name}'")
        wafer = np.load(request.app.state.samples_dir / f"{name}.npy", allow_pickle=False)
        return SampleWafer(name=name, wafer_map=wafer.tolist())

    @app.get("/past-wafers/{index}", response_model=PastWafer,
             responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
             tags=["samples"])
    def past_wafer(index: int, request: Request):
        store = request.app.state.past_store
        if store is None:
            raise HTTPException(status_code=503, detail="Past wafer maps are not available on this server")
        try:
            return store.get(index)
        except IndexError:
            raise HTTPException(status_code=404, detail=f"No past wafer #{index}")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()