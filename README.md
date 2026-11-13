# Anime Image Curator · 动漫图筛

本地化的文生图**质检 + 按 prompt 选优**工具，带 WebUI。**全程离线，不调用任何云端 AI。**

## 它解决什么

你用文生图批量出某个角色的同人图（一般同一段 prompt 出 4 张），人工逐张筛选费时费力。
本工具帮你把这件事变成「**每组确认一下推荐对不对**」：

1. 自动读取每张图里嵌入的 prompt（支持 SD-webui / ComfyUI，PNG 和 JPEG 都能读）；
2. 按 prompt 自动分组（同一段 prompt 的 4 张归一组）；
3. 给每张图打分（**质检 80% + 美观度 20% + 结构 5%**，权重可调；未启用项自动忽略、在已启用项间归一化），组内排序；糊图/过曝图额外硬降权；
4. 每组把「最优候选」顶到最前，你在 WebUI 里一眼扫、点选确认（也可以点「全部不选」跳过该组）；
5. 导出精选清单，或把每组精选直接拷贝到一个目录。

> **定位是「辅助分诊」，不是「全自动质检员」。** 数手指、判断服装结构这类难题，
> 本地没有能 100% 可靠的模型；工具会**标记可疑**并降权，但最终由你拍板。

## 能力分阶段（按需启用，越往下越重）

| 阶段 | 能力 | 依赖 | 说明 |
|---|---|---|---|
| **基础**（默认） | 读 prompt、分组、角色识别、轻量质检、WebUI、选优、导出 | 仅 fastapi/pillow 等 | 秒开，无大模型下载 |
| **+aesthetic** | 美观度打分（兼作结构合理性参考），**多模型可切换** | `--extra aesthetic` | 见下表，首次使用按需下载 |
| **+structure** | **结构打分（已转正，低权重参与排序）**：DWPose 133 关键点几何检查（躯体→手/手指→足/脚趾）+ 可选 YOLO 部位板块 | `--extra structure`（YOLO 另需 `--extra yolo`） | 详见 docs/STRUCTURE_SCORER.md |
| **+structure** | **角色还原（CCIP，影子收集数据中）**：图像侧身份特征比对，支持换装/换发型变体 | 同上 | 见「感知 v2」 |
| **+vlm** | **本地 VLM 解剖/角色判定**：便携 llama-server + Qwen3-VL-8B，零安装、自动启停 | 无额外 Python 依赖；`python scripts/setup_models.py --only vlm` | 见「本地 VLM」 |

## 感知 v2（2026-11）

结构/还原度判定全面升级，全部本地运行、零云端：

- **解剖检查细化到脚趾**：DWPose 133 关键点（身体+脸+双手+双脚）几何规则——
  躯体比例/关节反折 → 手指长度顺序/指节比例/指尖聚集 → 脚趾/脚跟几何（仅重度畸形扣分）。
  原则不变：**检出不到 = N/A 不扣分**，只对可见部位做检查。
- **人体检测器 A/B**：默认动漫域检测器（deepghs v1.1_m），HumanArt YOLOX-L 作影子对照，
  逐图记录分歧到 telemetry，用你的真实图库数据定最终选择（`structure.pose.detector` / `ab_shadow`）。
- **角色还原 CCIP（影子模式）**：图像侧身份特征比对，三层策略应对换装/换发型——
  参考图/变体最佳匹配（`characters.yaml` 配 `variants` + `reference_images`）、
  无参考图时组内动态锚定、组内质心离群只打 flag 不扣分。数据攒够再转正。
- **便携本地 VLM**：见下节。级联门控，只在可疑时调用。
- **全过程证据落盘**：`data/telemetry/run_*.jsonl`（结构子分、检测器 A/B、CCIP 距离、VLM 判定），
  供离线 replay 校准，见 `docs/FEEDBACK_FORMAT.md`。

一次性下载感知模型（DWPose/检测器/部位/CCIP，约 1GB）：

```bat
.venv\Scripts\python.exe scripts\setup_models.py --only pose,detector,parts,ccip
```

> 所有模型/二进制都在 `.models/` 与 HF 缓存内，不写注册表、不装服务、不改 PATH；删除即彻底清理。

## 美观度模型（WebUI 顶部下拉框可切换，切换后自动用新模型重打分）

| 模型 | 类型 | 体量 | 分数范围 | 说明 |
|---|---|---|---|---|
| **shadow-v2**（默认） | 动漫专用 | ~4.4GB | hq概率 0–1 | 1.1B ViT，1024×1024，danbooru 训练。社区公认最强的动漫美观度模型。原作者已下架，用备份仓库。 |
| **skytnt-anime** | 动漫专用 | ~112MB | 约 0–1 | ConvNeXtV2 ONNX，danbooru 训练。轻快，CPU/GPU 都快，适合大批量。 |
| **aesthetic-predictor-v2-5** | 通用+插画 | ~3.6GB | 1–10（5.5+优秀） | SigLIP，明确支持插画，社区生态最好、维护活跃。 |
| improved-clip | 通用(照片/AI图) | ~1GB | 约 1–10 | CLIP+MLP 基线，照片训练，动漫偏弱。 |

> 切换模型后用**你自己的图**对比分数——这是唯一可靠的检验，社区评价代替不了你的实际数据。
> 每个模型下载后缓存在 `.models/` 与 HF 缓存，之后无需重下。

> **网络慢 / 4.4GB 下不动？** 默认 shadow-v2 有 4.4GB，境内网络不稳时容易中断。
> 可先在下拉框切到 **skytnt-anime**（仅 112MB、同样动漫专用、CPU 也快），立刻就能用；
> 等网络好时再切回 shadow-v2。下载已做多源（hf-mirror→官方）+ 断点续传 + 重试，中断后重启会接着下。

## 快速开始

```bash
# Windows：双击 run.bat（或命令行）
run.bat
# Linux/WSL：
./run.sh
```

打开浏览器 → **http://127.0.0.1:8000** → 填入图片文件夹路径 → Scan。

> 如果端口被占用（如 8000 已在使用），会自动顺延到 8001、8002 等。

## 启动选项

`run.bat` 默认就会装上美观度模型依赖（含 torch）。首次运行稍慢，之后有缓存很快。

```bat
run.bat                :: 默认：含美观度模型（4个可切换）+ WebUI
run.bat --gpu          :: 同上，但 torch 用 NVIDIA CUDA 版（有 N 卡强烈推荐，快很多）
run.bat --base         :: 仅基础：秒开、不装 torch、无美观度模型
run.bat --folder "D:\imgs"   :: 启动后自动扫描某目录
run.bat --port 8080    :: 自定义端口（被占用时自动顺延）
```

> **有 NVIDIA 卡请用 `run.bat --gpu`**：默认装的是 CPU 版 torch，跑 4.4G 的 shadow-v2 会很慢；
> CUDA 版能用上你的显存，快几十倍。首次会下载 CUDA 版 torch（约 2.5GB，从 pytorch.org）。

## 本地 VLM（解剖合理性 + 角色清单判定）

运行时是**便携 llama.cpp llama-server（Vulkan）+ Qwen3-VL-8B abliterated GGUF**：
二进制与权重全在 `.models/` 内，零安装、零注册表、零服务、零 PATH 修改，不依赖 Ollama。

**1. 一次性下载（约 6.3GB：运行时 38MB + Q4_K_M 5.0GB + mmproj-f16 1.2GB）**

```bat
.venv\Scripts\python.exe scripts\setup_models.py --only vlm
```

境内默认走 hf-mirror（`AIC_HF_ENDPOINT` 可覆盖）；`--quant Q8_0` 可下更高精度做 A/B。

**2. 启用**：`config/curator.yaml`（从 example 复制）中设

```yaml
vlm_judge:
  enabled: true
```

**3. 正常启动应用即可（`run.bat`）——不需要手动开服务。**
首次有图触发判定时自动拉起 llama-server（127.0.0.1:8690，占用自动顺延；冷启动约 8s），
退出应用自动停止。日志在 `.models/vlm/server.log`。

- **级联门控省算力**：只在结构分低于 `gate`（默认 0.85）/有畸形 flag/需要角色清单时调用；
  `mode: all` 全量判定。每图细节裁剪（手优先，其次足/脸）最多 `max_crops` 个。
- **显存**：Q4_K_M + mmproj 约 6GB，8G 显存可跑；Vulkan 后端 N/A/I 卡通用。
- **复用已有端点**：设 `vlm_judge.base_url`（任何 OpenAI 兼容服务）则不启动自带服务。
- 权重 `weights.vlm_judge` 默认 0（先收集数据），校准后再转正。

## 配置

- `config/characters.example.yaml` — 角色 + 多套服装定义（名字 tag、LoRA、各服装 tag 集）。
  复制为 `config/characters.yaml` 后按需修改。每个角色可定义**多套官方皮肤/服装**，其中一套标 `default: true`。
- `config/curator.example.yaml` — 各打分项权重（默认质检 0.8、美观度 0.2、结构 0.05、VLM 0）、
  质检 flag 惩罚、组级弃选提示阈值、结构（检测器 A/B、足部阈值）、角色还原 CCIP、VLM 级联门控、缩略图尺寸等。
  复制为 `config/curator.yaml` 生效。

## 结构打分（已转正，需 `uv sync --extra structure`）

基于 **DWPose 133 关键点**（身体+脸+双手，onnxruntime，无 torch）的几何规则检查：
手指长度顺序/指节比例/指尖聚集/关节反折、左右肢体不对称、关节融合等。
可另配 **最多 4 个 YOLO 部位模型**（手/脚/脸…）作为辅助板块，板块分为各模型均分。

核心原则：**检出不到 = N/A 不扣分**（手放背后检不出手，不代表手有问题）；只对可见部位做几何检查。

- **2026-10 已转正**：依据 66 次导出会话 / 2173 组 / 8934 张图的反馈数据校准（structure 影子分与被选图
  Mann-Whitney z=4.69，组内 pairwise 胜率 57.5%），现以 `weights.structure: 0.05` 低权重参与排序。
  统计依据与方法见 **`docs/CALIBRATION_2026-10.md`**；`structure.shadow: true` 可退回纯影子模式。
- 配置见 `config/curator.example.yaml` 的 `structure:` 段；设计详见 `docs/STRUCTURE_SCORER.md`。
- YOLO 板块需 `uv sync --extra yolo`（引入 ultralytics/torch），并在 `structure.yolo.models` 配置。

## 反馈数据采集（校准结构打分的弹药）

导出区新增 **「记录本次导出」** 勾选框：勾选后每次导出（JSON/拷贝）都会把本次全部选择 + 所有打分特征追加到 `data/feedback/feedback-YYYY-MM-DD.jsonl`。感知 v2 的运行证据（结构子分、检测器 A/B 分歧、CCIP 距离、VLM 判定）另落 `data/telemetry/run_*.jsonl`。数据格式规格见 **`docs/FEEDBACK_FORMAT.md`**（含样本定义、N/A 语义、阈值校准与影子分转正的统计标准）。积累一段时间后把该目录交给分析（人或 AI）即可校准权重——首批校准（2026-10，structure 转正 + 质检惩罚 + 弃选提示）的过程与结论见 **`docs/CALIBRATION_2026-10.md`**。

## 元数据格式（已实测）

- **JPEG**：prompt 存在 EXIF `UserComment`，编码为 **UTF-16-BE**（很多工具读不出来，这里已处理）。
- **PNG (SD-webui)**：`parameters` 文本块。
- **PNG (ComfyUI 原生)**：`prompt`/`workflow` JSON（尽力解析）。
- 区块结构：`正面 prompt` → `Negative prompt:` → `Steps:..., Seed:..., Model:..., Version: ...`。
- 角色识别：名字 tag（如 `velina airgid`）+ LoRA（如 `<lora:Char-ZZZ-Velina-V1-IL:0.8>`）双重匹配。

## 目录

```
curator/
  metadata.py     prompt 解析（PNG/JPEG，含 UTF-16 UserComment）
  characters.py   角色/服装配置与匹配
  pipeline.py     扫描 + 分组 + 打分编排
  scorers/        可插拔打分器（quality / aesthetic / structure / character_ccip / vlm_judge）
                  + person_detect / dwpose_infer / ccip_infer / vlm_server / ort_utils / telemetry
  app.py          FastAPI + WebUI（启动打印各打分项可用性）
  static/index.html
scripts/setup_models.py  一次性模型下载（pose/detector/parts/ccip/vlm）
config/           配置示例
docs/             设计/校准/反馈格式文档
```

## 隐私

所有分析在本地完成。不上传任何图片或 prompt。

## 键盘快捷键

图片放大查看时（点击任意图片缩略图）：

| 按键 | 功能 |
|---|---|
| ← / → | 同组内切换上一张/下一张 |
| ↑ / ↓ | 切换到上一组/下一组（每组从第一张开始） |
| Enter | 选择当前查看的图片 |
| O | 叠加/取消 DWPose 骨架（需 structure 打分器） |
| Esc | 关闭放大视图 |

> 每组末尾有一张虚线卡片「⊘ 全部不选」，点击后该组不导出任何图片。

## 故障排查

- **美观度“未启用 / 加载失败：Can't load image processor ... preprocessor_config.json”**：
  这是模型文件没下全/网络中断的报错（transformers 会把下载失败误报成“找不到配置文件”）。
  现在下载层已改为**多源（hf-mirror→官方）+ 断点续传 + 重试**，并从本地缓存目录加载，重启应用会自动接着下。
  若 4.4GB 的 shadow-v2 总是下不动，先在下拉框切到 **skytnt-anime**（112MB）应急。
- **下载很慢**：确认 `run.bat`/`run.sh` 里 `HF_ENDPOINT=https://hf-mirror.com` 生效；有 N 卡用 `run.bat --gpu`。
- **新克隆后 UI 开关变灰/选框全是「未安装」**：venv 缺 extras。用 `run.bat` 启动（默认同步全部感知依赖），
  或手动 `uv sync --inexact --extra aesthetic --extra structure --extra yolo --group dev`
  后 `uv run --no-sync python -m curator.app`（裸 `uv run` 会精确重同步、卸掉 extras）。
  启动时 cmd 打印 `[scorer] OK/OFF 名称: 原因`，缺什么一眼可见；页面接口异常会显式红字报错。
