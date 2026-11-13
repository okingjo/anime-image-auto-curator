# Anime Image Curator

**English** · [中文](README.md)

A fully local **quality-check + per-prompt pick-the-best** tool for text-to-image batches, with a WebUI. **100% offline — no cloud AI calls, ever.**

## What it does

You batch-generate fan art of a character with a text-to-image model (typically 4 images per prompt), and hand-picking the good ones is tedious.
This tool turns that into "**just confirm the recommendation for each group**":

1. Reads the prompt embedded in every image (SD-webui / ComfyUI; both PNG and JPEG);
2. Groups images by prompt (the 4 images from one prompt form a group);
3. Scores every image (**quality 80% + aesthetic 20% + structure 5%** by default, weights adjustable; disabled scorers are ignored and weights renormalized across the enabled ones), ranks within each group; blurry/overexposed images get an extra hard penalty;
4. Floats the best candidate to the front of each group — you skim and click to confirm in the WebUI (or click "Select none" to skip a group);
5. Export the pick list as JSON, or copy each group's pick straight into a folder.

> **This is an assisted-triage tool, not a fully automatic inspector.** Counting fingers or judging
> costume structure is hard; no local model is 100% reliable. The tool **flags suspects** and
> down-weights them — you make the final call.

## Capability tiers (enable as needed; heavier further down)

| Tier | Capability | Dependencies | Notes |
|---|---|---|---|
| **Base** (default) | Prompt reading, grouping, character ID, lightweight QC, WebUI, picking, export | fastapi/pillow only | Instant start, no large model downloads |
| **+aesthetic** | Aesthetic scoring (also a structure-plausibility reference), **multiple switchable models** | `--extra aesthetic` | See table below; downloaded on first use |
| **+structure** | **Structure scoring (promoted, low weight in ranking)**: DWPose 133-keypoint geometry checks (body → hands/fingers → feet/toes) + optional YOLO part panels | `--extra structure` (YOLO also needs `--extra yolo`) | See docs/STRUCTURE_SCORER.md |
| **+structure** | **Character fidelity (CCIP, shadow data collection)**: image-side identity feature comparison, supports outfit/hairstyle variants | same as above | See "Perception v2" |
| **+vlm** | **Local VLM anatomy/character judgments**: portable llama-server + Qwen3-VL-8B, zero install, auto start/stop | No extra Python deps; `python scripts/setup_models.py --only vlm` | See "Local VLM" |

## Perception v2 (2026-11)

Structure/fidelity judgment fully upgraded — all local, zero cloud:

- **Anatomy checks down to the toes**: DWPose 133 keypoints (body + face + both hands + both feet) with geometric rules —
  body proportions / joint reversal → finger length ordering / phalanx ratios / fingertip clustering → toe/heel geometry (only severe deformities are penalized).
  The principle is unchanged: **not detected = N/A, no penalty** — only visible parts are checked.
- **Person-detector A/B**: anime-domain detector by default (deepghs v1.1_m), HumanArt YOLOX-L as a shadow control;
  per-image disagreements are logged to telemetry so the final choice can be settled on your real library (`structure.pose.detector` / `ab_shadow`).
- **Character fidelity CCIP (shadow mode)**: image-side identity comparison with a three-layer strategy for outfit/hairstyle changes —
  best match against reference images/variants (`characters.yaml` with `variants` + `reference_images`),
  dynamic in-group anchoring when no references exist, centroid outliers flagged but not penalized. Promoted once enough data accumulates.
- **Portable local VLM**: see below. Cascade-gated — only called when suspicious.
- **Full evidence trail on disk**: `data/telemetry/run_*.jsonl` (structure sub-scores, detector A/B, CCIP distances, VLM verdicts)
  for offline replay calibration — see `docs/FEEDBACK_FORMAT.md`.

One-time download of the perception models (DWPose / detector / parts / CCIP, ~1GB):

```bat
.venv\Scripts\python.exe scripts\setup_models.py --only pose,detector,parts,ccip
```

> All models/binaries live in `.models/` and the HF cache — no registry writes, no services, no PATH changes; delete to clean up completely.

## Aesthetic models (switchable in the WebUI top bar; switching auto re-scores with the new model)

| Model | Type | Size | Score range | Notes |
|---|---|---|---|---|
| **shadow-v2** (default) | Anime-specific | ~4.4GB | hq probability 0–1 | 1.1B ViT, 1024×1024, danbooru-trained. Widely regarded as the strongest anime aesthetic model. Original delisted; a backup repo is used. |
| **skytnt-anime** | Anime-specific | ~112MB | ~0–1 | ConvNeXtV2 ONNX, danbooru-trained. Fast on CPU/GPU, great for large batches. |
| **aesthetic-predictor-v2-5** | General + illustration | ~3.6GB | 1–10 (5.5+ excellent) | SigLIP, explicit illustration support, best community ecosystem and actively maintained. |
| improved-clip | General (photos/AI images) | ~1GB | ~1–10 | CLIP+MLP baseline, photo-trained, weak on anime. |

> After switching models, compare scores **on your own images** — that's the only reliable test; community reputation can't replace your actual data.
> Each model is cached in `.models/` and the HF cache after download — no re-downloads.

> **Slow network / 4.4GB won't finish?** The default shadow-v2 is 4.4GB and can stall on unstable connections.
> Switch to **skytnt-anime** in the dropdown first (only 112MB, also anime-specific, fast even on CPU) and start working immediately;
> switch back to shadow-v2 when the network cooperates. Downloads are multi-source (hf-mirror → official) + resumable + retried — a restart continues where it left off.

## Quick start

```bash
# Windows: double-click run.bat (or run from a terminal)
run.bat
# Linux/WSL:
./run.sh
```

Open your browser → **http://127.0.0.1:8000** → enter the image folder path → Scan.

The UI language auto-selects from your browser language (Chinese/English); the **EN / 中文** button in the top bar switches at any time and the choice is remembered.

While scanning, a **live progress bar** sits at the top: phase (listing files / scoring images / grouping & ranking / writing evidence), processed/total count, elapsed time and ETA, and the file currently being processed; a hint appears if the first image is waiting on model loading. Auto-scans triggered by `run.bat --folder` can be attached to mid-flight by opening the page later.

> If the port is busy (e.g. 8000 already in use), it auto-increments to 8001, 8002, etc.

## Launch options

`run.bat` installs the aesthetic-model dependencies (including torch) by default. The first run is a bit slow; afterwards the cache makes it fast.

```bat
run.bat                :: default: aesthetic models (4 switchable) + WebUI
run.bat --gpu          :: same, but NVIDIA CUDA torch (strongly recommended with an N card — much faster)
run.bat --base         :: base only: instant start, no torch, no aesthetic models
run.bat --folder "D:\imgs"   :: auto-scan a folder on startup
run.bat --port 8080    :: custom port (auto-increments if busy)
```

> **With an NVIDIA card, use `run.bat --gpu`**: the default is CPU torch, which is very slow for the 4.4GB shadow-v2;
> the CUDA build uses your VRAM and is tens of times faster. The first run downloads CUDA torch (~2.5GB from pytorch.org).

## Local VLM (anatomy plausibility + character-checklist judgments)

The runtime is a **portable llama.cpp llama-server (Vulkan) + Qwen3-VL-8B abliterated GGUF**:
binary and weights all live inside `.models/` — zero install, zero registry, zero services, zero PATH changes, no Ollama dependency.

**1. One-time download (~6.3GB: runtime 38MB + Q4_K_M 5.0GB + mmproj-f16 1.2GB)**

```bat
.venv\Scripts\python.exe scripts\setup_models.py --only vlm
```

In mainland China the default endpoint is hf-mirror (`AIC_HF_ENDPOINT` overrides); `--quant Q8_0` downloads a higher-precision variant for A/B tests.

**2. Enable**: in `config/curator.yaml` (copy from the example) set

```yaml
vlm_judge:
  enabled: true
```

**3. Just start the app normally (`run.bat`) — no manual server needed.**
On the first image that triggers a judgment, llama-server starts automatically (127.0.0.1:8690, auto-increments if busy; cold start ~8s),
and stops when the app exits. Log: `.models/vlm/server.log`.

- **Cascade gating saves compute**: called only when the structure score is below `gate` (default 0.85), deformity flags exist, or a character checklist is needed;
  `mode: all` judges every image. Per-image detail crops (hands first, then feet/face), at most `max_crops`.
- **VRAM**: Q4_K_M + mmproj ≈ 6GB — runs on an 8GB card; the Vulkan backend works on both NVIDIA and AMD.
- **Reuse an existing endpoint**: set `vlm_judge.base_url` (any OpenAI-compatible server) and the bundled server won't start.
- `weights.vlm_judge` defaults to 0 (data collection first); promote after calibration.

## Configuration

- `config/characters.example.yaml` — characters + multiple outfit definitions (name tags, LoRAs, per-outfit tag sets).
  Copy to `config/characters.yaml` and edit. Each character can define **multiple official skins/outfits**, one marked `default: true`.
- `config/curator.example.yaml` — scorer weights (default quality 0.8, aesthetic 0.2, structure 0.05, VLM 0),
  QC flag penalties, group-level skip-suggestion thresholds, structure (detector A/B, foot thresholds), character-fidelity CCIP, VLM cascade gating, thumbnail size, etc.
  Copy to `config/curator.yaml` to take effect.

## Structure scoring (promoted; requires `uv sync --extra structure`)

Geometric rule checks on **DWPose 133 keypoints** (body + face + both hands; onnxruntime, no torch):
finger length ordering / phalanx ratios / fingertip clustering / joint reversal, left-right limb asymmetry, fused joints, etc.
Optionally add **up to 4 YOLO part models** (hands/feet/face…) as auxiliary panels; the panel score is the mean across models.

Core principle: **not detected = N/A, no penalty** (hands behind the back aren't detected — that doesn't mean they're wrong); geometry checks run only on visible parts.

- **Promoted in 2026-10**: calibrated on feedback from 66 export sessions / 2173 groups / 8934 images (structure shadow score vs. selected images:
  Mann-Whitney z=4.69, in-group pairwise win rate 57.5%); now participates in ranking at the low weight `weights.structure: 0.05`.
  Statistics and method: **`docs/CALIBRATION_2026-10.md`**; `structure.shadow: true` reverts to pure shadow mode.
- Config: the `structure:` section of `config/curator.example.yaml`; design details: `docs/STRUCTURE_SCORER.md`.
- YOLO panels need `uv sync --extra yolo` (pulls ultralytics/torch) and entries in `structure.yolo.models`.

> Note: the per-image diagnostic flag texts (e.g. joint-angle warnings) are currently Chinese-only; an English pass is planned. Everything else in the UI is fully bilingual.

## Feedback collection (fuel for calibrating the structure score)

The export area has a **"Record this export"** checkbox: when checked, every export (JSON/copy) appends all selections + all scoring features to `data/feedback/feedback-YYYY-MM-DD.jsonl`. Perception-v2 run evidence (structure sub-scores, detector A/B disagreements, CCIP distances, VLM verdicts) goes to `data/telemetry/run_*.jsonl`. Format spec: **`docs/FEEDBACK_FORMAT.md`** (sample definitions, N/A semantics, threshold-calibration and shadow-promotion statistical criteria). After accumulating data, hand the directory to an analyst (human or AI) to calibrate weights — the first calibration round (2026-10: structure promotion + QC penalties + skip suggestions) is documented in **`docs/CALIBRATION_2026-10.md`**.

## Metadata formats (field-tested)

- **JPEG**: prompt in EXIF `UserComment`, encoded **UTF-16-BE** (many tools can't read it — handled here).
- **PNG (SD-webui)**: `parameters` text chunk.
- **PNG (ComfyUI native)**: `prompt`/`workflow` JSON (best-effort parsing).
- Block structure: `positive prompt` → `Negative prompt:` → `Steps:..., Seed:..., Model:..., Version: ...`.
- Character ID: dual matching on name tags (e.g. `velina airgid`) + LoRAs (e.g. `<lora:Char-ZZZ-Velina-V1-IL:0.8>`).

## Repository layout

```
curator/
  metadata.py     prompt parsing (PNG/JPEG, incl. UTF-16 UserComment)
  characters.py   character/outfit config and matching
  pipeline.py     scan + grouping + scoring orchestration
  scorers/        pluggable scorers (quality / aesthetic / structure / character_ccip / vlm_judge)
                  + person_detect / dwpose_infer / ccip_infer / vlm_server / ort_utils / telemetry
  app.py          FastAPI + WebUI (prints scorer availability on startup)
  static/index.html
scripts/setup_models.py  one-time model downloads (pose/detector/parts/ccip/vlm)
config/           example configs
docs/             design / calibration / feedback-format docs
```

## Privacy

All analysis happens locally. No images or prompts are ever uploaded.

## Keyboard shortcuts

While viewing an image full-size (click any thumbnail):

| Key | Action |
|---|---|
| ← / → | Previous / next image within the group |
| ↑ / ↓ | Previous / next group (starts at the first image) |
| Enter | Select the image currently shown |
| O | Toggle the DWPose skeleton overlay (requires the structure scorer) |
| Esc | Close the full-size view |

> Each group ends with a dashed "⊘ Select none" card — clicking it exports nothing from that group.

## Troubleshooting

- **Aesthetic "off / load failed: Can't load image processor ... preprocessor_config.json"**:
  this means model files didn't fully download / the network dropped (transformers misreports download failures as "config file not found").
  The download layer now uses **multi-source (hf-mirror → official) + resume + retry** and loads from the local cache directory; restarting the app continues the download.
  If the 4.4GB shadow-v2 keeps stalling, switch to **skytnt-anime** (112MB) in the dropdown as a stopgap.
- **Slow downloads**: make sure `HF_ENDPOINT=https://hf-mirror.com` is active in `run.bat`/`run.sh`; with an N card use `run.bat --gpu`.
- **Fresh clone: UI toggles greyed out / dropdown full of "not installed"**: the venv is missing extras. Start with `run.bat` (syncs all perception deps by default),
  or manually `uv sync --inexact --extra aesthetic --extra structure --extra yolo --group dev`
  then `uv run --no-sync python -m curator.app` (bare `uv run` re-syncs exactly and removes extras).
  Startup prints `[scorer] OK/OFF name: reason` in the cmd window so gaps are obvious; API failures show explicit red errors in the page.
