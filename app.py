#!/usr/bin/env python3
"""jh-super-resolution — Modal backend for JHenTai remote super-resolution.

One GPU container (scale-to-zero) serves SR jobs for exactly one fixed
model, RealESRGAN_x4plus_anime_6B (native 4x; 2x/3x are produced by a
Lanczos downscale of the 4x result).

Topology
--------
- Clients never talk to the modal.run URL. A cloudflared process inside
  the container exposes the HTTP server through a Cloudflare Tunnel
  under a per-account hostname (e.g. https://ehsr-1.fur.boats).
- The HTTP layer enforces Host strictly:
    Host == TUNNEL_HOST        -> client API (session-authenticated)
    Host == *.modal.run        -> only /__wake, /__session, /__status,
                                  and only with the X-Backend-Secret
                                  header (the scheduler's private door;
                                  hitting it is also what wakes the
                                  container)
    anything else              -> 403
- Sessions are minted only by the scheduler (POST /__session), so a
  client cannot bypass the scheduler with a made-up session id.

Job model
---------
A session receives one batch of work, three ways:
  POST /sr/submit-urls    {"urls": [...], "scale": 2|3|4}
                          Backend fetches each image itself. Original
                          image links (fullimg.php / or=...) are
                          rejected on purpose: SR of originals is not
                          offered, and those links need account auth
                          the backend does not have.
  POST /sr/submit-archive {"url": "...", "scale": 2|3|4}
                          Backend downloads the zip, extracts images,
                          upscales all of them, repacks a zip.
  POST /sr/begin-upload   {"total": N, "scale": 2|3|4}
  POST /sr/upload-one     multipart file + form field "name"
                          For galleries already downloaded on the
                          phone: the client uploads the local images.
Progress:  GET  /sr/status   -> stage, totals, per-session counts and
                                how many images are queued ahead.
Result:    GET  /sr/result   -> one zip with the upscaled images
                                (original file names preserved).
Cancel:    DELETE /sr/session

GPU admission: a global FIFO queue feeds a small worker pool; a worker
only starts an image when free VRAM is above a floor (or nothing else
is in flight, to avoid deadlock), so a burst of big pages queues
instead of OOMing the card. Busy GPU seconds are tallied per month on
a Volume and reported via /__status — that is what the scheduler's
"wake the account with the most credit left" decision reads.

Secrets (modal secret create jh-sr-secrets ...):
  BACKEND_SECRET   shared with the scheduler (required)
  TUNNEL_HOST      this account's tunnel hostname, no scheme
                   (e.g. ehsr-1.fur.boats)
  CF_TUNNEL_TOKEN  Cloudflare Tunnel token (from the Zero Trust
                   dashboard; the tunnel's public hostname must route
                   to http://localhost:8080). Without it the server
                   still runs, reachable only via modal.run's private
                   door — clients will not be able to use it.

Deploy: see README.md in this repo.
"""

import io
import json
import os
import re
import secrets as pysecrets
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path

import modal

APP_NAME = "jh-super-resolution"
GPU_TYPE = "T4"
PORT = 8080
MODEL_PATH = "/opt/models/RealESRGAN_x4plus_anime_6B.pth"
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)
DATA_DIR = Path("/data")
SESSIONS_DIR = Path("/tmp/sessions")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MIN_FREE_VRAM = 3_200_000_000  # ~3.2 GB free required to admit a job
EST_JOB_VRAM = 2_600_000_000  # rough per-job reservation while running
MAX_WORKERS = 2
SESSION_TTL_SECONDS = 2 * 3600
RESULT_KEEP_SECONDS = 30 * 60

# ---------------------------------------------------------------------------
# Modal wiring
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .pip_install(
        "torch==2.3.1",
        "fastapi",
        "uvicorn",
        "pillow",
        "numpy",
        "httpx",
        "python-multipart",
    )
    .run_commands(
        "mkdir -p /opt/models"
        f" && curl -fL -o {MODEL_PATH} {MODEL_URL}"
        " && curl -fL -o /usr/local/bin/cloudflared"
        " https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        " && chmod +x /usr/local/bin/cloudflared"
    )
)

volume = modal.Volume.from_name("jh-sr-data", create_if_missing=True)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Vendored RRDBNet (the Real-ESRGAN network; architecture only, weights are
# loaded from the official release file at runtime)
# ---------------------------------------------------------------------------

def _rrdbnet_class():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            body = self.conv_body(self.body(feat))
            feat = feat + body
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    return RRDBNet


def load_model():
    import torch

    net = _rrdbnet_class()(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=6, num_grow_ch=32)
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("params_ema") or state.get("params") or state
    net.load_state_dict(state, strict=True)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda().half()
    return net


# ---------------------------------------------------------------------------
# Inference (tiled, VRAM-bounded)
# ---------------------------------------------------------------------------

TILE = 512
TILE_PAD = 16


def upscale_pil(img, scale, model):
    """Run the 4x model tile by tile; downscale to 2x/3x when asked."""
    import numpy as np
    import torch
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    src = np.asarray(img).astype(np.float32) / 255.0  # HWC
    out = np.zeros((h * 4, w * 4, 3), dtype=np.float32)
    device = next(model.parameters()).device

    with torch.no_grad():
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                cx0 = max(0, x0 - TILE_PAD)
                cy0 = max(0, y0 - TILE_PAD)
                cx1 = min(w, x0 + TILE + TILE_PAD)
                cy1 = min(h, y0 + TILE + TILE_PAD)
                crop = src[cy0:cy1, cx0:cx1]
                t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
                if device.type == "cuda":
                    t = t.half()
                sr = model(t).float().clamp_(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                # paste the non-padded region into the 4x canvas
                ox0, oy0 = x0 * 4, y0 * 4
                ox1, oy1 = min(w, x0 + TILE) * 4, min(h, y0 + TILE) * 4
                sx0, sy0 = (x0 - cx0) * 4, (y0 - cy0) * 4
                out[oy0:oy1, ox0:ox1] = sr[sy0 : sy0 + (oy1 - oy0), sx0 : sx0 + (ox1 - ox0)]
                del t, sr
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = Image.fromarray((out * 255.0).round().astype("uint8"))
    if scale != 4:
        result = result.resize((w * scale, h * scale), Image.LANCZOS)
    return result


def save_like(img, dest: Path, orig_suffix: str):
    suffix = orig_suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.save(dest, "JPEG", quality=95)
    elif suffix == ".webp":
        img.save(dest, "WEBP", quality=95)
    elif suffix == ".bmp":
        img.save(dest, "BMP")
    else:
        img.save(dest, "PNG")


# ---------------------------------------------------------------------------
# Usage accounting (busy GPU seconds per month, persisted on the Volume)
# ---------------------------------------------------------------------------


class UsageMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.file = DATA_DIR / "usage.json"
        self.month = time.strftime("%Y-%m")
        self.seconds = 0.0
        self._load()

    def _load(self):
        try:
            data = json.loads(self.file.read_text())
            if data.get("month") == self.month:
                self.seconds = float(data.get("gpuSeconds", 0))
        except Exception:
            pass

    def add(self, seconds: float):
        with self.lock:
            month = time.strftime("%Y-%m")
            if month != self.month:
                self.month = month
                self.seconds = 0.0
            self.seconds += seconds

    def snapshot(self) -> float:
        with self.lock:
            if time.strftime("%Y-%m") != self.month:
                return 0.0
            return self.seconds

    def persist(self):
        with self.lock:
            payload = {"month": self.month, "gpuSeconds": self.seconds}
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(payload))
            volume.commit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Job manager: sessions + one global FIFO queue + VRAM-gated workers
# ---------------------------------------------------------------------------


class Session:
    def __init__(self, sid: str):
        self.sid = sid
        self.created_at = time.time()
        self.dir = SESSIONS_DIR / sid
        (self.dir / "out").mkdir(parents=True, exist_ok=True)
        self.mode = None  # 'urls' | 'archive' | 'upload'
        self.scale = 4
        self.submitted = False
        self.intake_done = False
        self.intake_error = None
        self.expected_total = None  # upload mode: declared by the client
        self.received = 0
        self.items = {}  # idx -> {"name", "kind": url|file, "src", "seq"}
        self.done = {}  # idx -> output filename
        self.errors = {}  # idx -> message
        self.result_ready_at = None
        self.downloaded_at = None
        self.lock = threading.Lock()

    def total(self):
        if self.mode == "upload":
            return self.expected_total if self.expected_total is not None else len(self.items)
        return len(self.items)

    def is_complete(self):
        if not self.submitted or not self.intake_done:
            return False
        total = self.total()
        return total > 0 and len(self.done) + len(self.errors) >= total


class JobManager:
    def __init__(self, model):
        self.model = model
        self.sessions = {}
        self.sessions_lock = threading.Lock()
        self.queue = deque()  # (seq, sid, idx)
        self.queue_cond = threading.Condition()
        self.seq = 0
        self.inflight = 0
        self.inflight_lock = threading.Lock()
        self.usage = UsageMeter()
        self.workers = []
        self._stop = False

    # -- sessions ------------------------------------------------------
    def create_session(self) -> str:
        sid = pysecrets.token_urlsafe(24)
        with self.sessions_lock:
            self.sessions[sid] = Session(sid)
        return sid

    def get_session(self, sid):
        with self.sessions_lock:
            return self.sessions.get(sid)

    def drop_session(self, sid):
        with self.sessions_lock:
            session = self.sessions.pop(sid, None)
        if session is not None:
            with self.queue_cond:
                self.queue = deque(e for e in self.queue if e[1] != sid)
                self.queue_cond.notify_all()
            shutil.rmtree(session.dir, ignore_errors=True)

    # -- intake --------------------------------------------------------
    def enqueue(self, session: Session, idx: int, item: dict):
        with self.queue_cond:
            self.seq += 1
            item = {**item, "seq": self.seq}
            session.items[idx] = item
            self.queue.append((self.seq, session.sid, idx))
            self.queue_cond.notify()

    def queued_ahead(self, session: Session) -> int:
        """Images from other sessions still waiting ahead of this one."""
        with self.queue_cond:
            pending = list(self.queue)
        my_seqs = [seq for seq, sid, _ in pending if sid == session.sid]
        if not my_seqs:
            return 0
        first = min(my_seqs)
        return sum(1 for seq, sid, _ in pending if sid != session.sid and seq < first)

    def processing_count(self, session: Session) -> int:
        return sum(
            1
            for idx, item in session.items.items()
            if item.get("processing") and idx not in session.done and idx not in session.errors
        )

    # -- workers -------------------------------------------------------
    def start(self):
        for i in range(MAX_WORKERS):
            t = threading.Thread(target=self._worker_loop, name=f"sr-worker-{i}", daemon=True)
            t.start()
            self.workers.append(t)
        threading.Thread(target=self._usage_loop, name="usage", daemon=True).start()
        threading.Thread(target=self._reaper_loop, name="reaper", daemon=True).start()

    def _usage_loop(self):
        last = time.time()
        while not self._stop:
            time.sleep(5)
            now = time.time()
            with self.inflight_lock:
                busy = self.inflight > 0
            with self.queue_cond:
                busy = busy or len(self.queue) > 0
            if busy:
                self.usage.add(now - last)
            last = now
            if int(now) % 30 < 5:
                self.usage.persist()

    def _reaper_loop(self):
        while not self._stop:
            time.sleep(300)
            now = time.time()
            with self.sessions_lock:
                sids = list(self.sessions.keys())
            for sid in sids:
                session = self.get_session(sid)
                if session is None:
                    continue
                expired = now - session.created_at > SESSION_TTL_SECONDS
                drained = (
                    session.downloaded_at is not None
                    and now - session.downloaded_at > RESULT_KEEP_SECONDS
                )
                if expired or drained:
                    self.drop_session(sid)

    def _vram_admission(self) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return True
            free, _total = torch.cuda.mem_get_info()
            with self.inflight_lock:
                inflight = self.inflight
            return free - EST_JOB_VRAM * inflight >= MIN_FREE_VRAM or inflight == 0
        except Exception:
            return True

    def _worker_loop(self):
        while not self._stop:
            with self.queue_cond:
                while not self.queue:
                    self.queue_cond.wait(timeout=5)
                    if self._stop:
                        return
                entry = self.queue[0]
            if not self._vram_admission():
                time.sleep(1)
                continue
            with self.queue_cond:
                if not self.queue or self.queue[0] != entry:
                    continue
                self.queue.popleft()
            _seq, sid, idx = entry
            session = self.get_session(sid)
            if session is None:
                continue
            item = session.items.get(idx)
            if item is None:
                continue
            item["processing"] = True
            with self.inflight_lock:
                self.inflight += 1
            try:
                self._process_item(session, idx, item)
            except Exception as e:  # keep the worker alive no matter what
                session.errors[idx] = str(e)
            finally:
                item["processing"] = False
                with self.inflight_lock:
                    self.inflight -= 1
                if session.is_complete():
                    threading.Thread(target=self._pack_result, args=(session,), daemon=True).start()

    def _process_item(self, session: Session, idx: int, item: dict):
        import httpx
        from PIL import Image

        if item["kind"] == "url":
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                resp = client.get(item["src"])
                resp.raise_for_status()
                raw = resp.content
            src_path = session.dir / f"in_{idx}"
            src_path.write_bytes(raw)
        else:
            src_path = Path(item["src"])

        img = Image.open(src_path)
        img.load()
        result = upscale_pil(img, session.scale, self.model)
        out_name = item["name"]
        dest = session.dir / "out" / out_name
        save_like(result, dest, Path(out_name).suffix)
        session.done[idx] = out_name

    def _pack_result(self, session: Session):
        if session.result_ready_at is not None:
            return
        zip_path = session.dir / "result.zip"
        tmp_path = session.dir / "result.zip.tmp"
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx in sorted(session.done.keys()):
                name = session.done[idx]
                zf.write(session.dir / "out" / name, arcname=name)
            if session.errors:
                lines = [f"{session.items[i]['name']}: {msg}" for i, msg in sorted(session.errors.items())]
                zf.writestr("_errors.txt", "\n".join(lines))
        tmp_path.rename(zip_path)
        session.result_ready_at = time.time()

    def stage_of(self, session: Session) -> str:
        if session.intake_error:
            return "error"
        if not session.submitted or not session.intake_done:
            return "receiving"
        if session.result_ready_at is not None:
            return "done"
        if session.is_complete():
            return "packing"
        if self.processing_count(session) > 0 or session.done or session.errors:
            return "processing"
        return "queued"


# ---------------------------------------------------------------------------
# HTTP layer (FastAPI). RUNTIME is set by Backend.start() inside the
# container; the FastAPI app itself is importable anywhere.
# ---------------------------------------------------------------------------

RUNTIME = {"manager": None}

TUNNEL_HOST = os.environ.get("TUNNEL_HOST", "").strip()
BACKEND_SECRET = os.environ.get("BACKEND_SECRET", "")

ORIGINAL_LINK_RE = re.compile(r"fullimg|or=", re.IGNORECASE)


def build_fastapi_app():
    from fastapi import FastAPI, File, Form, Header, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

    api = FastAPI(title=APP_NAME, docs_url=None, redoc_url=None, openapi_url=None)

    def manager() -> JobManager:
        return RUNTIME["manager"]

    def fail(status: int, message: str):
        return JSONResponse({"error": message}, status_code=status)

    def check_secret(request: Request):
        if not BACKEND_SECRET:
            return fail(500, "BACKEND_SECRET is not configured on this backend")
        if request.headers.get("x-backend-secret") != BACKEND_SECRET:
            return fail(401, "bad backend secret")
        return None

    def get_session_or_404(sid):
        session = manager().get_session(sid) if sid else None
        return session

    @api.middleware("http")
    async def host_guard(request: Request, call_next):
        host = (request.headers.get("host") or "").split(":")[0].lower()
        path = request.url.path
        if TUNNEL_HOST and host == TUNNEL_HOST.lower():
            return await call_next(request)
        if host.endswith(".modal.run"):
            if path in ("/__wake", "/__session", "/__status"):
                return await call_next(request)
            return fail(403, "this host only serves the scheduler door")
        return fail(403, "host not allowed")

    # -- public ---------------------------------------------------------
    @api.get("/health")
    async def health():
        return {"ok": True}

    # -- scheduler door (modal.run host + secret) ------------------------
    @api.post("/__wake")
    async def wake(request: Request):
        err = check_secret(request)
        if err:
            return err
        ensure_cloudflared()
        return {"ok": True, "tunnelHost": TUNNEL_HOST}

    @api.post("/__session")
    async def mint_session(request: Request):
        err = check_secret(request)
        if err:
            return err
        return {"sessionId": manager().create_session()}

    @api.get("/__status")
    async def status(request: Request):
        err = check_secret(request)
        if err:
            return err
        m = manager()
        with m.inflight_lock:
            inflight = m.inflight
        with m.queue_cond:
            queued = len(m.queue)
        return {
            "ok": True,
            "busy": inflight > 0 or queued > 0,
            "inflight": inflight,
            "queued": queued,
            "sessions": len(m.sessions),
            "gpuSecondsThisMonth": m.usage.snapshot(),
        }

    # -- client API (tunnel host + session) ------------------------------
    def session_from(request: Request):
        sid = request.headers.get("x-session-id", "")
        return get_session_or_404(sid)

    @api.post("/sr/submit-urls")
    async def submit_urls(request: Request):
        session = session_from(request)
        if session is None:
            return fail(401, "unknown session")
        if session.submitted:
            return fail(409, "session already has a batch")
        body = await request.json()
        urls = body.get("urls") or []
        scale = int(body.get("scale") or 4)
        if scale not in (2, 3, 4):
            return fail(400, "scale must be 2, 3 or 4")
        accepted, rejected = [], 0
        for u in urls:
            if not isinstance(u, str) or not u.startswith("http"):
                rejected += 1
            elif ORIGINAL_LINK_RE.search(u):
                rejected += 1  # originals are not accepted for SR
            else:
                accepted.append(u)
        if not accepted:
            return fail(400, "no acceptable image urls (original-image links are rejected)")
        session.mode = "urls"
        session.scale = scale
        session.submitted = True
        session.intake_done = True
        for i, u in enumerate(accepted):
            name = Path(u.split("?")[0]).name or f"image_{i}.jpg"
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                name += ".jpg"
            manager().enqueue(session, i, {"name": name, "kind": "url", "src": u})
        return {"accepted": len(accepted), "rejected": rejected, "total": len(accepted)}

    @api.post("/sr/submit-archive")
    async def submit_archive(request: Request):
        session = session_from(request)
        if session is None:
            return fail(401, "unknown session")
        if session.submitted:
            return fail(409, "session already has a batch")
        body = await request.json()
        url = body.get("url") or ""
        scale = int(body.get("scale") or 4)
        if scale not in (2, 3, 4):
            return fail(400, "scale must be 2, 3 or 4")
        if not url.startswith("http"):
            return fail(400, "archive url required")
        session.mode = "archive"
        session.scale = scale
        session.submitted = True

        def intake():
            import httpx
            from PIL import Image

            try:
                zip_path = session.dir / "archive.zip"
                with httpx.Client(follow_redirects=True, timeout=600) as client:
                    with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        with open(zip_path, "wb") as f:
                            for chunk in resp.iter_bytes(1 << 20):
                                f.write(chunk)
                extract_dir = session.dir / "extracted"
                extract_dir.mkdir(exist_ok=True)
                idx = 0
                with zipfile.ZipFile(zip_path) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        name = Path(info.filename).name
                        if Path(name).suffix.lower() not in IMAGE_EXTS:
                            continue
                        target = extract_dir / f"{idx}_{name}"
                        with zf.open(info) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        # verify it really is an image before queueing
                        with Image.open(target) as im:
                            im.verify()
                        manager().enqueue(session, idx, {"name": name, "kind": "file", "src": str(target)})
                        idx += 1
                if idx == 0:
                    session.intake_error = "archive contains no images"
            except Exception as e:
                session.intake_error = f"archive intake failed: {e}"
            finally:
                session.intake_done = True

        threading.Thread(target=intake, daemon=True).start()
        return {"accepted": True, "note": "archive is being fetched; poll /sr/status"}

    @api.post("/sr/begin-upload")
    async def begin_upload(request: Request):
        session = session_from(request)
        if session is None:
            return fail(401, "unknown session")
        if session.submitted:
            return fail(409, "session already has a batch")
        body = await request.json()
        total = int(body.get("total") or 0)
        scale = int(body.get("scale") or 4)
        if total <= 0:
            return fail(400, "total must be positive")
        if scale not in (2, 3, 4):
            return fail(400, "scale must be 2, 3 or 4")
        session.mode = "upload"
        session.scale = scale
        session.expected_total = total
        session.submitted = True
        session.intake_done = True
        return {"accepted": True, "total": total}

    @api.post("/sr/upload-one")
    async def upload_one(request: Request, file: UploadFile = File(...), name: str = Form(...)):
        session = session_from(request)
        if session is None:
            return fail(401, "unknown session")
        if session.mode != "upload" or not session.submitted:
            return fail(409, "call /sr/begin-upload first")
        with session.lock:
            idx = session.received
            if session.expected_total is not None and idx >= session.expected_total:
                return fail(409, "upload exceeds the declared total")
            session.received += 1
        safe_name = Path(name).name or f"image_{idx}.jpg"
        dest = session.dir / f"up_{idx}_{safe_name}"
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        manager().enqueue(session, idx, {"name": safe_name, "kind": "file", "src": str(dest)})
        return {"received": session.received, "total": session.expected_total}

    @api.get("/sr/status")
    async def sr_status(request: Request):
        session = session_from(request)
        if session is None:
            return fail(401, "unknown session")
        m = manager()
        payload = {
            "stage": m.stage_of(session),
            "total": session.total(),
            "done": len(session.done),
            "failed": len(session.errors),
            "processing": m.processing_count(session),
            "queuedAhead": m.queued_ahead(session),
        }
        if session.intake_error:
            payload["error"] = session.intake_error
        return payload

    @api.get("/sr/result")
    async def sr_result(request: Request):
        session = session_from(request)
        if session is None:
            return fail(401, "unknown session")
        zip_path = session.dir / "result.zip"
        if session.result_ready_at is None or not zip_path.exists():
            return fail(409, "result is not ready yet")
        session.downloaded_at = time.time()
        return FileResponse(zip_path, media_type="application/zip", filename="super_resolution.zip")

    @api.delete("/sr/session")
    async def sr_cancel(request: Request):
        session = session_from(request)
        if session is None:
            return fail(401, "unknown session")
        manager().drop_session(session.sid)
        return {"ok": True}

    return api


fastapi_app = None


def get_fastapi_app():
    """Build the FastAPI app lazily, inside the container.

    `modal deploy` executes this module locally first, and the local
    environment does not have the container image's packages (fastapi
    and friends) installed — building the app at import time would
    crash the deploy before it even starts.
    """
    global fastapi_app
    if fastapi_app is None:
        fastapi_app = build_fastapi_app()
    return fastapi_app

# ---------------------------------------------------------------------------
# cloudflared (started next to the HTTP server inside the container)
# ---------------------------------------------------------------------------

_cloudflared_proc = None
_cloudflared_lock = threading.Lock()


def ensure_cloudflared():
    global _cloudflared_proc
    token = os.environ.get("CF_TUNNEL_TOKEN", "").strip()
    if not token:
        return
    with _cloudflared_lock:
        if _cloudflared_proc is not None and _cloudflared_proc.poll() is None:
            return
        _cloudflared_proc = subprocess.Popen(
            [
                "cloudflared",
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://localhost:{PORT}",
                "run",
                "--token",
                token,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------


@app.cls(
    gpu=GPU_TYPE,
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("jh-sr-secrets")],
    scaledown_window=300,
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
class Backend:
    @modal.enter()
    def start(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        model = load_model()
        manager = JobManager(model)
        manager.start()
        RUNTIME["manager"] = manager
        ensure_cloudflared()

    @modal.web_server(PORT, startup_timeout=300)
    def serve(self):
        import socket

        def run():
            import uvicorn

            uvicorn.run(get_fastapi_app(), host="0.0.0.0", port=PORT, log_level="info")

        threading.Thread(target=run, name="uvicorn", daemon=True).start()
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", PORT), timeout=2):
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError("uvicorn did not start in time")
