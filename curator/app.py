"""FastAPI app + WebUI for the Anime Image Curator."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .pipeline import Curator

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"

app = FastAPI(title="Anime Image Curator")
curator = Curator()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_path(path: str) -> Path:
    """Only allow files under the scanned root (prevent path traversal)."""
    if not curator.root:
        raise HTTPException(400, "尚未扫描任何目录")
    p = Path(path).resolve()
    root = Path(curator.root).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(403, "路径不在扫描目录内")
    if not p.is_file():
        raise HTTPException(404, "文件不存在")
    return p


def _thumb(path: Path, size: int) -> Path:
    cache = _ROOT / ".thumbs"
    cache.mkdir(exist_ok=True)
    key = hashlib.sha1(f"{path}|{path.stat().st_mtime}|{size}".encode()).hexdigest()
    out = cache / f"{key}.jpg"
    if not out.exists():
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size))
            img.save(out, "JPEG", quality=82)
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/status")
def status():
    return {
        "root": curator.root,
        "scorers": [
            {"name": s.name, "label": s.label, "available": s.available,
             "reason": getattr(s, "reason", "")}
            for s in curator.scorers.values()
        ],
        "weights": curator.weights,
        "characters": list(curator.characters.keys()),
        "aesthetic_backends": curator.aesthetic_backends(),
        "scanned": bool(curator.groups),
    }


@app.get("/api/aesthetic_backends")
def aesthetic_backends():
    return {"backends": curator.aesthetic_backends()}


class SetAestheticReq(BaseModel):
    backend: str


@app.post("/api/set_aesthetic")
def set_aesthetic(req: SetAestheticReq):
    res = curator.set_aesthetic_backend(req.backend)
    if not res.get("ok"):
        raise HTTPException(400, res.get("reason", "切换失败"))
    return res


class ScanReq(BaseModel):
    folder: str
    recursive: bool | None = None


@app.post("/api/scan")
def scan(req: ScanReq):
    summary = curator.scan(req.folder, req.recursive)
    return summary


@app.get("/api/state")
def state():
    return curator.state()


@app.get("/api/image")
def image(path: str = Query(...)):
    return FileResponse(_safe_path(path))


@app.get("/api/thumb")
def thumb(path: str = Query(...), size: int = Query(512)):
    size = max(128, min(1600, size))
    return FileResponse(_thumb(_safe_path(path), size), media_type="image/jpeg")


class SelectReq(BaseModel):
    key: str
    filename: str | None = None


@app.post("/api/select")
def select(req: SelectReq):
    ok = curator.set_selection(req.key, req.filename)
    if not ok:
        raise HTTPException(404, "分组不存在")
    return {"ok": True}


@app.get("/api/selected")
def selected():
    return {"selected": curator.selected_paths()}


class ExportReq(BaseModel):
    mode: str = "json"   # json | copy
    dest: str | None = None


@app.post("/api/export")
def export(req: ExportReq):
    if req.mode == "copy":
        if not req.dest:
            raise HTTPException(400, "copy 模式需要 dest 目录")
        copied = curator.export_copy(req.dest)
        return {"mode": "copy", "copied": copied}
    return JSONResponse({"mode": "json", "selected": curator.selected_paths()})


# --------------------------------------------------------------------------- #
def main():
    # 防御性默认：未设置 HF_ENDPOINT 时（如双击旧版 run.bat）用国内镜像，
    # 避免访问不可达的 huggingface.co。（代码层还有 endpoint 回退兼底。）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    parser = argparse.ArgumentParser(description="Anime Image Curator")
    parser.add_argument("--folder", help="启动后自动扫描的目录")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config", help="curator.yaml 路径")
    args = parser.parse_args()

    if args.config:
        global curator
        curator = Curator(args.config)

    # Eagerly load heavy models in the background (non-blocking).
    curator.prewarm()

    if args.folder:
        # Non-blocking: a first-time model download must not prevent the
        # server from starting. Scan runs in a background thread.
        import threading

        def _bg_scan(folder):
            try:
                print("[INFO] auto-scan:", folder, flush=True)
                print(curator.scan(folder), flush=True)
            except Exception as e:
                print("[WARN] auto-scan failed:", e, flush=True)

        threading.Thread(target=_bg_scan, args=(args.folder,), daemon=True).start()

    import uvicorn

    print("=" * 60)
    print("Anime Image Curator")
    print(f"Open http://{args.host}:{args.port}")
    print("=" * 60)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
