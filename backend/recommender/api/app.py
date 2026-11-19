"""The web API. Run it from the backend folder:

    uvicorn recommender.api.app:app --reload

Then open http://127.0.0.1:8000/docs to try it.

Errors are answered in plain words: unknown investor 404, bad input 422.
An unexpected failure is logged and answered with a short 500, never a stack trace.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from recommender.api.service import MAX_K, RecommendationService, UnknownInvestor

log = logging.getLogger("recommender.api")


def create_app(service: RecommendationService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or RecommendationService()   # builds once, at startup
        yield

    app = FastAPI(title="Startup recommender", version="1", lifespan=lifespan,
                  description="Picks from the current champion model, each with an exact why.")
    # The demo page runs on the developer's own machine (Vite's default port)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET"], allow_headers=["*"])

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        log.exception("Unexpected error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Something went wrong on our side. It has been logged."})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/champion")
    def champion(request: Request):
        return request.app.state.service.champion

    @app.get("/investors")
    def investors(request: Request, search: str = Query("", max_length=100),
                  limit: int = Query(20, ge=1, le=100)):
        return {"investors": request.app.state.service.search(search, limit)}

    @app.get("/picks")
    def picks(request: Request, investor: str = Query(..., min_length=1, max_length=200),
              k: int = Query(10, ge=1, le=MAX_K)):
        try:
            return request.app.state.service.picks(investor, k)
        except UnknownInvestor:
            raise HTTPException(status_code=404, detail=(
                f"No investor {investor!r} with at least 5 past deals before the cutoff. "
                "Use /investors?search=... to find a valid id."))

    return app


app = create_app()
