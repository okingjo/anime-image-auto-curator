"""Self-contained local VLM runtime: portable llama.cpp ``llama-server``.

Zero system pollution by design:
  * runtime binaries + GGUF weights live entirely in ``.models/vlm/``
  * the server is a child process started on demand and stopped on shutdown
  * no installer, no registry, no service, no PATH changes — deleting
    ``.models/vlm`` removes every trace.

Bootstrap (once):  ``python scripts/setup_models.py --only vlm``
Then the curator starts/stops the server automatically. An existing
OpenAI-compatible endpoint (Ollama / vLLM / llama-swap) can be used instead
via ``vlm_judge.base_url`` — nothing is spawned in that case.
"""
from __future__ import annotations

import atexit
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
VLM_DIR = _ROOT / ".models" / "vlm"
PATHS_JSON = VLM_DIR / "vlm_paths.json"

DEFAULT_PORT = 8690


def load_paths() -> dict:
    for p in (PATHS_JSON, _ROOT / ".models" / "vlm_paths.json"):
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                # normalize key schema (setup_models writes server_exe/gguf)
                if "server_exe" in d and "server" not in d:
                    d["server"] = d["server_exe"]
                if "gguf" in d and "model" not in d:
                    d["model"] = d["gguf"]
                return d
            except Exception:
                pass
    return {}


def _free_port(preferred: int = DEFAULT_PORT) -> int:
    for port in (preferred, preferred + 1, preferred + 2, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def _health(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health",
                                    timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class LlamaServer:
    """Lifecycle manager for the portable llama-server child process."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self._lock = threading.RLock()  # reentrant: ensure_running() calls stop() while holding it
        self.last_error = ""
        atexit.register(self.stop)

    # -- state ---------------------------------------------------------------
    @property
    def base_url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.port else None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def installed(self) -> bool:
        p = load_paths()
        return bool(p.get("server") and Path(p["server"]).exists()
                    and p.get("model") and Path(p["model"]).exists()
                    and p.get("mmproj") and Path(p["mmproj"]).exists())

    # -- lifecycle -----------------------------------------------------------
    def ensure_running(self, ctx: int = 8192, parallel: int = 2,
                       timeout_s: float = 240.0) -> str:
        """Start the server if needed; return its base URL. Raises on failure."""
        with self._lock:
            if self.running() and self.base_url and _health(self.base_url):
                return self.base_url
            self.stop()
            paths = load_paths()
            server = paths.get("server", "")
            model = paths.get("model", "")
            mmproj = paths.get("mmproj", "")
            if not (server and model and mmproj):
                raise RuntimeError(
                    "VLM 未下载。运行: python scripts/setup_models.py --only vlm")
            for p in (server, model, mmproj):
                if not Path(p).exists():
                    raise RuntimeError(f"VLM 文件缺失: {p}（重新运行 setup_models.py）")
            self.port = _free_port()
            cmd = [
                server,
                "-m", model,
                "--mmproj", mmproj,
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "-c", str(int(ctx)),
                "--parallel", str(int(parallel)),
                "-ngl", "99",
                "--flash-attn", "on",
                "--no-webui",
                "--alias", "curator-vlm",
            ]
            log_path = VLM_DIR / "server.log"
            try:
                VLM_DIR.mkdir(parents=True, exist_ok=True)
                logf = open(log_path, "ab")
            except OSError:
                logf = subprocess.DEVNULL
            flags = 0
            if sys.platform == "win32":
                flags = subprocess.CREATE_NO_WINDOW
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=logf, stderr=subprocess.STDOUT,
                    cwd=str(Path(server).parent), creationflags=flags)
            except Exception as e:
                self.last_error = f"启动失败: {e}"
                raise RuntimeError(self.last_error)
            deadline = time.time() + timeout_s
            url = self.base_url
            assert url is not None
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    tail = ""
                    try:
                        tail = log_path.read_text(encoding="utf-8",
                                                  errors="replace")[-600:]
                    except OSError:
                        pass
                    self.last_error = f"llama-server 提前退出: {tail}"
                    raise RuntimeError(self.last_error)
                if _health(url):
                    return url
                time.sleep(1.0)
            self.stop()
            self.last_error = f"llama-server {timeout_s:.0f}s 内未就绪（见 .models/vlm/server.log）"
            raise RuntimeError(self.last_error)

    def stop(self):
        with self._lock:
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                except Exception:
                    pass
                self.proc = None
            self.port = None


# module-level singleton shared by all scorers
SERVER = LlamaServer()
