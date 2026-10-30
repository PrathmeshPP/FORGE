"""
F.O.R.G.E. Website Backend — Main Application Entry Point
FastAPI application with unified single-origin static file serving and strict security headers.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from web_backend.config import HOST, PORT, FRONTEND_DIST_DIR, FORGE_DATA_DIR
from web_backend.database import init_db
from web_backend.api.routes import router


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds mandatory security headers to every HTTP response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Ensure database schema is initialized on startup
    FORGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="F.O.R.G.E. Indoor 3D Reconstruction Platform",
    version="1.0.0",
    docs_url=None, # Disabled per SECURITY_BASELINE to prevent operational leakage
    redoc_url=None,
    lifespan=lifespan
)

# Apply Security Headers
app.add_middleware(SecurityHeadersMiddleware)

# Custom Exception Handler to enforce safe JSON errors (no raw tracebacks)
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {"code": "HTTP_ERROR", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log unexpected errors locally, return safe generic message to client
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected workstation error occurred."}
    )


# Mount API Router
app.include_router(router)

# Unified Single-Origin Serving: Serve React Production Build
if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
    # Mount assets folder
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Mount sample folder
    sample_dir = FRONTEND_DIST_DIR / "sample"
    if sample_dir.exists():
        app.mount("/sample", StaticFiles(directory=str(sample_dir)), name="sample")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Do not intercept API requests
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Endpoint not found."})

        target_file = FRONTEND_DIST_DIR / full_path
        if target_file.is_file() and full_path != "":
            return FileResponse(str(target_file))

        # Default to SPA index.html
        return FileResponse(str(FRONTEND_DIST_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    # Bind exclusively to 127.0.0.1
    uvicorn.run("web_backend.main:app", host=HOST, port=PORT, reload=False)

