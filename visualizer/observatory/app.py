"""SKIRTGPU Observatory web app. The command-line launcher defaults to loopback."""

import os
import json
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog import Catalog
from .kinematics import availability as velocity_availability
from .jobs import ARTIFACTS, Jobs, execution_log, tail
from .simulation import BANDS, QUALITY, RenderRequest
from .spectra import SPECTRAL_MODES, MIN_RESOLVING_POWER, MAX_RESOLVING_POWER, reference_redshift
from .preview import PreviewRequest, PreviewVolumes
from .backend_preview import BackendPreview, BackendFrame


ROOT = Path(__file__).resolve().parents[1]


def preview_options(red: str = 'SLOAN_SDSS_I', green: str = 'SLOAN_SDSS_R', blue: str = 'SLOAN_SDSS_G',
                    size: int = 256, span: float = 60, x: float = 0, y: float = 0, z: float = 0, kernel: float = .5):
    # Parse query integers before validating the discrete grid-size choices.
    try:
        return PreviewRequest(red=red, green=green, blue=blue, size=size, span=span, x=x, y=y, z=z, kernel=kernel)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def create_app(config=None, storage=None, executable=None, allowed_hosts=None):
    config = Path(config or os.environ.get("SKIRT_VIEWER_DATASETS", ROOT / "datasets.json")).resolve()
    storage = Path(storage or os.environ.get("SKIRT_VIEWER_STORAGE", ROOT / "var")).resolve()
    executable = Path(executable or os.environ.get("SKIRT_VIEWER_BINARY", ROOT.parents[1] / "release/SKIRT/main/skirtgpu")).resolve()
    catalog = Catalog(config, storage / "catalogues")
    previews = PreviewVolumes(catalog)
    backend_preview = BackendPreview(storage,previews.root)

    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = Jobs(catalog, storage / "jobs", executable,
                              threads=int(os.environ.get("SKIRT_VIEWER_THREADS", "16")),
                              dust_batch_cells=(int(os.environ["SKIRT_VIEWER_DUST_BATCH_CELLS"])
                                                if os.environ.get("SKIRT_VIEWER_DUST_BATCH_CELLS") else None))
        yield
        app.state.jobs.close()
        backend_preview.close()

    app = FastAPI(title="SKIRTGPU Observatory", lifespan=lifespan)
    app.state.backend_preview = backend_preview
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts or ["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site" or (origin and urlsplit(origin).netloc != request.headers.get("host")):
                return JSONResponse({"detail": "Cross-origin writes are not permitted"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/api/config")
    def get_config():
        return {"datasets": [{**d.info(), 'source_velocities': velocity_availability(d)} for d in catalog.datasets.values()], "bands": BANDS, "quality": QUALITY, "spectral_modes": SPECTRAL_MODES,
                "spectral_resolution": {"min": MIN_RESOLVING_POWER, "max": MAX_RESOLVING_POWER, "samples_per_resolution_element": 2},
                "backend": {"available": executable.is_file() and os.access(executable, os.X_OK), "executable": str(executable)},
                "camera_model": "Distant observer / orthographic", "default_dataset": next(iter(catalog.datasets))}

    @app.get("/api/datasets/{identifier}/cloud")
    def get_cloud(identifier: str):
        if identifier not in catalog.datasets:
            raise HTTPException(404, "Unknown dataset")
        try:
            _, meta = catalog.cloud(identifier)
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error
        return {**meta, "layers": [{**layer, "url": f"/api/datasets/{identifier}/cloud/{layer['file']}?v={meta['fingerprint']}"} for layer in meta["layers"]]}

    @app.get("/api/datasets/{identifier}/cloud/{filename}")
    def cloud_file(identifier: str, filename: str):
        if identifier not in catalog.datasets:
            raise HTTPException(404, "Unknown dataset")
        folder, meta = catalog.cloud(identifier)
        if filename not in {layer["file"] for layer in meta["layers"]}:
            raise HTTPException(404, "Unknown catalogue layer")
        return FileResponse(folder / filename, media_type="application/octet-stream")

    @app.get('/api/datasets/{identifier}/preview-volume')
    def preview_volume(identifier: str, options: PreviewRequest = Depends(preview_options)):
        if identifier not in catalog.datasets:
            raise HTTPException(404, 'Unknown dataset')
        try:
            return previews.build(identifier, options)
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get('/api/preview-volumes/{key}/{filename}')
    def preview_file(key: str, filename: str):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key) or filename not in ('stars-dust.bin', 'young.bin'):
            raise HTTPException(404, 'Unknown preview volume')
        folder = previews.root/key
        if not (folder/'volume.json').is_file() or not (folder/filename).is_file():
            raise HTTPException(404, 'Preview volume is not ready')
        return FileResponse(folder/filename, media_type='application/octet-stream')

    @app.get('/api/backend-preview/status')
    def backend_status(request: Request):
        return {**backend_preview.status(request),'preparing_volume':previews.lock.locked()}

    @app.post('/api/backend-preview/sessions')
    def backend_session(request: Request):
        return JSONResponse(backend_preview.create_session(request),headers={'Cache-Control':'no-store'})

    @app.post('/api/backend-preview/frame')
    def backend_frame(payload: BackendFrame, request: Request):
        return Response(backend_preview.frame(request,payload),media_type='application/octet-stream',
                        headers={'Cache-Control':'no-store'})

    @app.delete('/api/backend-preview/sessions/{identifier}')
    def backend_release(identifier: str, request: Request):
        backend_preview.release(request,identifier)
        return {'released':True}

    @app.post("/api/jobs", status_code=202)
    def submit_job(payload: RenderRequest, request: Request):
        if payload.dataset_id not in catalog.datasets:
            raise HTTPException(422, "Unknown dataset or filter")
        try:
            job, reused = request.app.state.jobs.submit(payload)
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error
        return {"job": job, "reused": reused}

    def job_or_404(request, identifier):
        try:
            return request.app.state.jobs.get(identifier)
        except KeyError as error:
            raise HTTPException(404, "Unknown observation") from error

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        return {"jobs": request.app.state.jobs.list()}

    @app.get("/api/jobs/{identifier}")
    def get_job(identifier: str, request: Request):
        return job_or_404(request, identifier)

    @app.post("/api/jobs/{identifier}/cancel")
    def cancel_job(identifier: str, request: Request):
        job_or_404(request, identifier)
        return request.app.state.jobs.cancel(identifier)

    @app.get("/api/jobs/{identifier}/log")
    def get_log(identifier: str, request: Request):
        job_or_404(request, identifier)
        return {"text": tail(execution_log(storage / "jobs" / identifier))}

    @app.get("/api/jobs/{identifier}/spectrum")
    def get_spectrum(identifier: str, request: Request):
        job = job_or_404(request, identifier)
        path = storage / "jobs" / identifier / "spectrum.json"
        instruments = job["request"].get("instruments") or [job["request"].get("output", "image")]
        if job["status"] != "completed" or "sed" not in instruments or not path.is_file():
            raise HTTPException(404, "Spectrum is not available")
        payload = json.loads(path.read_text())
        # Enrich old observations without rewriting their saved science products.
        payload['reference_redshift'] = reference_redshift(path.parent/'observation.ski', payload.get('spectral_frame'))
        return payload

    @app.get("/api/jobs/{identifier}/files/{filename}")
    def get_artifact(identifier: str, filename: str, request: Request):
        job_or_404(request, identifier)
        if filename not in ARTIFACTS:
            raise HTTPException(404, "Unknown artifact")
        path = storage / "jobs" / identifier / filename
        if not path.is_file():
            raise HTTPException(404, "Artifact is not available yet")
        display_images = {"preview.png", "rgb.png", "spectrum.png"}
        return FileResponse(path, filename=None if filename in display_images else filename)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app
