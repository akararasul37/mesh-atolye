import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import FRONTEND, JOBS, ROOT, capabilities

MAX_UPLOAD = 20 * 1024 * 1024
JOB_FILES_LOCK = threading.RLock()
Image.MAX_IMAGE_PIXELS = 32_000_000
FINAL = {"done", "failed"}
STAGES = {
    "queued": "Sırada",
    "background": "Görsel hazırlanıyor",
    "views": "Ön, yan ve arka görünüş ayrılıyor",
    "shape_load": "Geometri modeli yükleniyor",
    "shape": "Geometri üretiliyor",
    "surface": "Yüzey ağı çıkarılıyor",
    "shape_done": "Geometri tamamlandı",
    "paint_load": "PBR modeli yükleniyor",
    "uv": "UV ve kamera görünümleri hazırlanıyor",
    "paint": "PBR kaplamaları üretiliyor",
    "bake": "Kaplama yüzeye işleniyor",
    "export": "Dosyalar hazırlanıyor",
    "paint_done": "Kaplama tamamlandı",
    "reference": "Kaynak görüntüler yüzeye aktarılıyor",
    "reference_done": "Kaynak kaplaması tamamlandı",
    "retry": "Düşük VRAM ayarlarıyla yeniden çalışıyor",
    "done": "Model hazır",
    "failed": "Üretim tamamlanamadı",
}


class StageError(RuntimeError):
    def __init__(self, stage, exit_code, details):
        super().__init__(f"{stage} süreci kapandı (kod {exit_code}).")
        self.stage = stage
        self.exit_code = exit_code
        self.details = details


class Jobs:
    def __init__(self):
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation")
        self.process = None
        self.stopping = False
        JOBS.mkdir(exist_ok=True)
        for job in self.list():
            if job["status"] not in FINAL:
                self.update(
                    JOBS / job["id"], status="failed", stage="failed",
                    error="Uygulama üretim sırasında kapandı. Görseli yeniden gönderin.",
                )

    @staticmethod
    def read(folder: Path):
        try:
            with JOB_FILES_LOCK:
                return json.loads((folder / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def write(folder, job):
        temporary = folder / "job.tmp"
        with JOB_FILES_LOCK:
            temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, folder / "job.json")

    def list(self):
        return sorted(
            (job for folder in JOBS.iterdir() if folder.is_dir() and (job := self.read(folder))),
            key=lambda job: job["created_at"], reverse=True,
        )

    def update(self, folder, **changes):
        with self.lock:
            job = self.read(folder)
            if "progress" in changes:
                changes["progress"] = max(job["progress"], changes["progress"])
            job.update(changes)
            self.write(folder, job)

    def submit(self, image, filename, mode, quality, cleanup, layout, profile):
        job_id = str(uuid4())
        folder = JOBS / job_id
        folder.mkdir()
        image.save(folder / "source.png")
        thumbnail = image.copy()
        thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
        thumbnail.save(folder / "thumbnail.webp", "WEBP", quality=82)
        job = {
            "id": job_id,
            "name": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "queued", "stage": "queued", "progress": 0,
            "mode": mode, "quality": quality, "remove_background": cleanup,
            "layout": layout, "profile": profile,
            "seed": secrets.randbits(32),
        }
        return self.enqueue(folder, job)

    def submit_refinement(self, source, parent):
        job_id = str(uuid4())
        folder = JOBS / job_id
        folder.mkdir()
        files = (
            "source.png", "thumbnail.webp", "prepared.png", "views.json",
            "view-front.png", "view-left.png", "view-right.png", "view-back.png",
            "model.glb", "model.obj", "model.mtl", "model-obj.zip",
            "model.jpg", "model_metallic.jpg", "model_roughness.jpg",
        )
        for filename in files:
            path = source / filename
            if path.is_file():
                shutil.copyfile(path, folder / filename)
        job = {
            **{key: parent[key] for key in (
                "name", "mode", "remove_background", "layout", "profile", "seed", "input_layout", "views",
            ) if key in parent},
            "id": job_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "queued", "stage": "queued", "progress": 0, "quality": "reference",
            "refinement_of": parent["id"],
        }
        return self.enqueue(folder, job)

    def enqueue(self, folder, job):
        with self.lock:
            self.write(folder, job)
            self.executor.submit(self.run, folder, job)
        return job

    def log_stage(self, folder, stage, args):
        command = [sys.executable, "-u", "-m", "mesh_atolye.worker", stage, str(folder), *args]
        with self.lock:
            if self.stopping:
                raise RuntimeError("Uygulama kapatılıyor.")
            self.process = subprocess.Popen(
                command, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            process = self.process
        last_error = ""
        try:
            with process.stdout as stream, (folder / "generation.log").open("a", encoding="utf-8") as log:
                log.write(f"\n=== {stage} {' '.join(args)} ===\n")
                for line in stream:
                    log.write(line)
                    log.flush()
                    if line.startswith("MESH_EVENT "):
                        event = json.loads(line.removeprefix("MESH_EVENT "))
                        self.update(folder, **event)
                    elif line.startswith(("RuntimeError:", "ValueError:", "OSError:")):
                        last_error = line.partition(":")[2].strip()
            code = process.wait()
        finally:
            with self.lock:
                self.process = None
        if code:
            raise StageError(stage, code, last_error)

    def gpu_stage(self, folder, stage, args):
        try:
            self.log_stage(folder, stage, args)
        except StageError as exc:
            if exc.exit_code != 42:
                raise
            self.update(folder, stage="retry", memory_retry=True)
            self.log_stage(folder, stage, [*args, "--low-memory"])

    def run(self, folder, job):
        started = time.perf_counter()
        self.update(folder, status="running")
        try:
            if job.get("refinement_of"):
                self.gpu_stage(folder, "refine", [])
            else:
                cleanup = ["--layout", job["layout"], "--profile", job["profile"]]
                if job["remove_background"]:
                    cleanup.append("--remove-background")
                self.log_stage(folder, "prepare", cleanup)
                settings = ["--quality", job["quality"], "--seed", str(job["seed"])]
                self.gpu_stage(folder, "shape", settings)
                if job["mode"] == "pbr":
                    self.gpu_stage(folder, "paint", settings)
                    if job["quality"] in {"reference", "studio"}:
                        self.gpu_stage(folder, "refine", [])
                else:
                    self.update(folder, stage="export", progress=96)
                    (folder / "geometry.glb").replace(folder / "model.glb")
                    (folder / "geometry.obj").replace(folder / "model.obj")
            files = ["model.glb", "model.obj"]
            if job["mode"] == "pbr":
                files.append("model-obj.zip")
            if any(not (folder / file).is_file() or (folder / file).stat().st_size == 0 for file in files):
                raise RuntimeError("Üretimden eksik dosya döndü.")
            (folder / "paint-input.obj").unlink(missing_ok=True)
            self.update(folder, status="done", stage="done", progress=100, glb_bytes=(folder / "model.glb").stat().st_size,
                        elapsed_seconds=round(time.perf_counter() - started, 1))
        except Exception as exc:
            message = str(exc)
            if isinstance(exc, StageError):
                if exc.exit_code == 42:
                    message = "CUDA belleği doldu. GPU kullanan diğer uygulamaları kapatıp tekrar deneyin."
                elif exc.details:
                    message = exc.details[:400]
            with (folder / "generation.log").open("a", encoding="utf-8") as log:
                log.write(f"\n{type(exc).__name__}: {message}\n")
            self.update(folder, status="failed", stage="failed", error=message)

    def stop(self):
        with self.lock:
            self.stopping = True
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        self.executor.shutdown(wait=True, cancel_futures=True)


@asynccontextmanager
async def lifespan(app):
    manager = Jobs()
    app.state.jobs = manager
    try:
        yield
    finally:
        manager.stop()


app = FastAPI(title="Mesh Atölye", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


def can_refine(folder, job):
    return (
        job["mode"] == "pbr" and job["status"] == "done" and not job.get("reference_applied")
        and (folder / "model.jpg").is_file()
        and ((folder / "prepared.png").is_file() or (folder / "view-front.png").is_file())
    )


def public(job):
    return {**job, "label": STAGES[job["stage"]], "can_refine": can_refine(JOBS / job["id"], job)}


def find_job(job_id):
    folder = JOBS / str(job_id)
    job = Jobs.read(folder)
    if job is None:
        raise HTTPException(404, "Üretim bulunamadı.")
    return folder, job


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return capabilities()


def require_local_origin(request):
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "İstek uygulamanın yerel adresinden gelmeli.")


@app.post("/generate", status_code=202)
def generate(
    request: Request,
    image: Annotated[UploadFile, File()],
    mode: Annotated[Literal["geometry", "pbr"], Form()] = "pbr",
    quality: Annotated[Literal["fast", "quality", "reference", "studio"], Form()] = "quality",
    remove_background: Annotated[bool, Form()] = True,
    layout: Annotated[Literal["auto", "single", "turnaround"], Form()] = "single",
    profile: Annotated[Literal["left", "right"], Form()] = "right",
):
    require_local_origin(request)
    installed = capabilities()
    if not installed.get(mode):
        raise HTTPException(503, "Seçilen model hazır değil. kurulum.bat dosyasını çalıştırın.")
    if layout == "turnaround" and not installed["multiview"]:
        raise HTTPException(503, "Çok görünüş modeli eksik. kurulum.bat dosyasını çalıştırın.")
    if image.size is not None and image.size > MAX_UPLOAD:
        raise HTTPException(413, "Görsel en fazla 20 MB olabilir.")
    raw = image.file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Görsel en fazla 20 MB olabilir.")
    try:
        with Image.open(BytesIO(raw), formats=["PNG", "JPEG", "WEBP"]) as source:
            if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("pixel limit")
            image_data = ImageOps.exif_transpose(source).convert("RGBA")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise HTTPException(422, "32 megapikseli aşmayan geçerli bir PNG, JPEG veya WebP görseli yükleyin.")
    name = (image.filename or "Görsel").replace("\\", "/").split("/")[-1][:120]
    return public(request.app.state.jobs.submit(image_data, name, mode, quality, remove_background, layout, profile))


@app.get("/api/jobs")
def history(request: Request, offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=100)] = 24):
    jobs = request.app.state.jobs.list()
    return {"items": [public(job) for job in jobs[offset:offset + limit]], "total": len(jobs)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: UUID):
    return public(find_job(job_id)[1])


@app.post("/api/jobs/{job_id}/refine", status_code=202)
def refine_job(job_id: UUID, request: Request):
    require_local_origin(request)
    folder, job = find_job(job_id)
    if not capabilities().get("pbr"):
        raise HTTPException(503, "PBR motoru hazır değil. kurulum.bat dosyasını çalıştırın.")
    if not can_refine(folder, job):
        raise HTTPException(409, "Bu kayıt kaynak görüntülerle yeniden kaplanamıyor.")
    return public(request.app.state.jobs.submit_refinement(folder, job))


@app.get("/api/jobs/{job_id}/thumbnail")
def thumbnail(job_id: UUID):
    folder, _ = find_job(job_id)
    return FileResponse(folder / "thumbnail.webp", media_type="image/webp")


@app.get("/api/jobs/{job_id}/source")
def get_source(job_id: UUID):
    folder, _ = find_job(job_id)
    return FileResponse(folder / "source.png", media_type="image/png")


@app.get("/api/jobs/{job_id}/views/{view}")
def get_view(job_id: UUID, view: Literal["front", "left", "right", "back"]):
    folder, job = find_job(job_id)
    if view not in job.get("views", ()):
        raise HTTPException(404, "Görünüş bulunamadı.")
    return FileResponse(folder / f"view-{view}.png", media_type="image/png")


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: UUID):
    folder, _ = find_job(job_id)
    path = folder / "generation.log"
    if not path.is_file():
        raise HTTPException(404, "Üretim günlüğü henüz oluşmadı.")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.get("/api/jobs/{job_id}/preview")
@app.get("/api/jobs/{job_id}/download/{form}")
def download(job_id: UUID, form: Literal["glb", "obj", "obj-zip"] | None = None):
    folder, job = find_job(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "Üretim henüz hazır değil.")
    names = {"glb": ("model.glb", "model/gltf-binary"), "obj": ("model.obj", "text/plain"),
             "obj-zip": ("model-obj.zip", "application/zip")}
    name, mime = names[form or "glb"]
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, "Dosya bulunamadı.")
    return FileResponse(path, media_type=mime, filename=f"mesh-{str(job_id)[:8]}-{name}" if form else None)
