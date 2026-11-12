# 感知层升级方案：解剖合理性 → 角色还原度 → 美观度

> 2026-11 调查与方案（v2，按用户反馈修订）。
> 优先级：**① 躯体到细节（手指/脚趾）的合理性判定最大化 → ② 角色还原度 → ③ 美观度。**
> 全程 SFW 纯技术质检框架（助手侧表述约束；VLM 侧直接用拒答消融模型，不做复杂防拒答机制）。
> 所有模型数据均经核实（HF 模型卡 / llama.cpp releases / rtmlib 官方模型表 / 实测 API），来源见文末。

---

## 1. 根因诊断（已读代码确认，不是权重问题）

| 症状 | 根因 | 代码位置 |
|---|---|---|
| VLM 一次都没激活 | `vlm_judge.py` 是空壳：`available = False`，"Phase 3 打分逻辑开发中"，`score()` 恒返回占位；配置 `weights.vlm_judge: 0.0`。**从未实现，改 yaml 无效** | `curator/scorers/vlm_judge.py` |
| 四肢/手指经常误判 | 结构分完全依赖 DWPose：`yolox_l.onnx` 是 **COCO 真人**训练的检测器，对动漫人体框不准；133 关键点从 **288×384 全身小图**回归，手部仅几十像素 → 21 个手部关键点本质是噪声，在其上跑几何规则 → 误报+漏报 | `dwpose_infer.py`、`structure.py` |
| 畸形"完全不识别" | 几何规则只在关键点置信度达标时生效；畸形手恰好产生低置信关键点 → **N/A 不扣分**；手掌 < 人物框高 3.5% 也 N/A。YOLO 部位板块 `models: []` 为空。**没有任何信号源覆盖手以外细节** | `structure.py` `_check_hand` |
| 脚趾完全没查 | DWPose 有脚部关键点（17-22：大趾/小趾/脚跟×2），代码只用 `feet_conf` 做可见性统计，**零条脚趾规则** | `structure.py` `_check_body` |
| 特殊部位畸形不识别 | 无检测器/规则覆盖躯干细节；本该 VLM 兜底，但 VLM 是空壳 | 同上 |
| 角色还原"不识别" | `characters.py` 只匹配 **prompt 文本**（name_tags + LoRA 文件名），不看像素。prompt 没写角色名 → 无身份；LoRA 触发但角色画崩 → 照样满分。outfit_ratio 也只是 prompt tag 重叠率 | `curator/characters.py` |
| 调权重没用 | 加权平均无硬门禁：美观分高可埋掉结构问题。感知层没有的信号，融合层调不出来 | `pipeline.py` |

**结论：问题在感知层能力缺失，不在融合层。方案 = 升级检测器 + 实现 VLM + 图像侧角色识别。**

---

## 2. 技术调查结论（已核实）

### 2.1 VLM 判定引擎：Qwen3-VL（便携 llama.cpp，零安装）

- Qwen3-VL-8B-Instruct：8.8B，Apache-2.0，原生 256K ctx；DeepStack 多层 ViT 融合（细粒度细节是这代主升级）；官方模型卡明确宣称 anime 识别增强。
- 显存：BF16 ≈ 17.6GB / FP8 ≈ 8.8GB / INT4 ≈ 5GB；KV ≈ 144KiB/token（fp16）。
- **运行时（已落地，替代 Ollama）**：便携 `llama-server.exe`（llama.cpp b12322，Windows Vulkan 构建，~38MB）+ GGUF 权重，全部放在 `.models/` 内，由 `curator/scorers/vlm_server.py` 按需启停子进程（OpenAI 兼容 HTTP，默认端口 8690）。**零系统安装、零注册表、零服务、不改 PATH，删目录即净**；不依赖 Ollama，也不依赖任何远端推理。`vlm_judge.base_url` 可复用既有 OpenAI 兼容端点（此时不 spawn）。
- 权重：`mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF` Q4_K_M 5.03GB + mmproj-f16 1.16GB（`scripts/setup_models.py --only vlm` 下载，hf-mirror 直连）；`--quant Q8_0` 作后续精度 A/B 档。
- **默认用拒答消融版**（abliteration 只处理文本部分、视觉能力不变）。用户已有消融模型路线，VLM 拒答不作为风险项；仅保留 JSON 解析失败兜底（工程必需）。
- 本机 RTX 5090 Laptop 24GB：Q4_K_M + 视觉塔 + 8K ctx 实测可启动（Vulkan），与其余 ONNX 模型共存无压力。

### 2.2 人体检测器：两个候选，P0 实测 A/B 决定

**候选 A：YOLOX-HumanArt（drop-in，零代码改动）**
- 现用 `yolox_l.onnx` = COCO 真人训练。rtmlib 官方注明：**"Models trained on HumanArt can detect both real human and cartoon characters. Models trained on COCO can only detect real human."**
- HumanArt 数据集（CVPR 2023）：50,000 图 / 20 场景（5 自然 + 3 二维人工〔动漫/卡通等〕+ 12 三维人工），标注含人体框 + 21 关键点 + 自接触关键点；检测器为 COCO+HumanArt 联合训练的 YOLOX。
- rtmlib 官方模型表 AP(person)：COCO 训练 yolox_l **50.1** → HumanArt+COCO 训练 yolox_m **59.1** / l **60.2** / x **61.3**（官方表未注明评测集，同口径 person AP）。
- 同 YOLOX 架构同输出格式 → `dwpose_infer.py` 只换模型文件，后处理零改动。
- 下载：`download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a62.zip`（m/x 同目录）。

**候选 B：deepghs/anime_person_detection（纯动漫专用，domain 更对口）**
- YOLOv8 n/s/m/x，训练集 = AniDet3（16k 动画剧集）+ 自有 v1.1（9.2k 插画/漫画，Roboflow 标注）；官方数据集卡注明两者混合训练是 best practice（纯 AniDet3 对插画/漫画弱）。
- 官方基准：**person_detect_v1.1_m F1 0.87**（阈值 0.348）；v1.2_s / v1.3_s F1 0.86。ONNX 现成，`imgutils.detect_person(image, level='m', version='v1.1')` 一行调用。
- 对"纯动漫插画"这一目标域，比 HumanArt（自然+人工混合）更专门化，**很可能是更优终点**。

**结论**：HumanArt 是稳妥的即插即用升级，但不是天花板。P0 在用户实际图库上 A/B 两者（检出率 + 框质量，100 张样本），胜者上线；架构上检测器与 DWPose 解耦（只喂人体框），可随时换。
**注意**：检测器只解决"框得准"；手指细节仍靠 crop 放大 + 关键点重估/VLM，与检测器选择无关。
- 备选：rtmlib 模型库还有 **RTMDet-nano 手部检测器**（AP hand 76.0，320×320，5 数据集训练），可作 ADetailer 手模型的对照。

### 2.3 手/脸专用检测：ADetailer YOLO 板块

- `Bingsu/adetailer`：`hand_yolov8n.pt` mAP50 **0.767**、`hand_yolov9c.pt` **0.810**（训练集含 AnHDet、hand-detection-fuao9 动漫手数据）；`face_yolov8n_v2.pt` / `face_yolov8s.pt`。
- 已知短板：部分绘画图上手模型置信度偏低、偶有误检 → `vis_floor` 降至 ~0.25 + 与 DWPose 腕部关键点框**取并集**兜底；仍不够换 `styler00dollar/YOLO-anime-hands`（YOLOv8x/v9e，17k 动漫手）或 RTMDet-nano。
- 仓库 YOLO 板块机制（`structure.yolo.models`，`repo:file` 自动下载）**已为此设计好，只是没填模型**。

### 2.4 图像侧角色识别：CCIP（deepghs）

- CCIP = Contrastive Character-Image Pretraining，回答"两张动漫图是否同一角色"（0-shot 相似度，非分类器，不认识角色名）。
- `deepghs/ccip_onnx`，ONNX（384×384 输入，768 维特征），`dghs-imgutils` 调用，**免 torch**：
  `ccip_extract_feature / ccip_difference / ccip_batch_differences / ccip_default_threshold`。
- 官方基准（"同角色"为正例）：

  | 模型 | F1 | Precision | Recall | 差异阈值 |
  |---|---|---|---|---|
  | **ccip-caformer_b36-24** | **0.941** | 0.938 | 0.944 | 0.213 |
  | ccip-caformer-24-randaug-pruned（imgutils 默认） | 0.917 | 0.933 | 0.901 | 0.178 |

- 官方明示局限：**只支持单人图**（多人需逐人 crop）；**发型捕捉强、发色/肤色敏感度弱**；需要参考图补偿。
- 换装鲁棒性：训练数据 zerochan 同人图本身含服装变化；新版数据集 merged_v1（33k 角色 / 114 万图）**明确把 Alter/泳装/职阶变更等变体合并为同一角色**（性别转换保持分离）——换装鲁棒是官方演进方向。**发型改变仍是已知强敏感项** → 用 §6 的分组设计解决，不硬扛阈值。

### 2.5 远程算力现状（实测）

- ai-server01 `qwen3.8-27b-uncensored:latest`：`/api/show` → `capabilities: [tools, thinking, completion]`，**无 vision**，不能当 VLM。
- 远程 VLM 选项：ai-server01（5×3090，Linux）拉 `huihui_ai/qwen3-vl-abliterated:8b-instruct-q8_0`，或上 **vLLM + Qwen3-VL-8B**（3090 无 FP8 硬件 → BF16 17.6GB 单卡可容，或社区 AWQ INT4）吃连续批处理吞吐。curator 端只改 `base_url`。

---

## 3. 目标管线（六级）

```
Stage 0  元数据/分组                        [现有，不动]
Stage 1  检测层（快，全图）
         ├─ 人体框：HumanArt YOLOX 或 deepghs anime YOLOv8（P0 A/B）
         ├─ DWPose 133 关键点               [现有，检测器升级后更准]
         └─ ADetailer YOLO：hand / face     [启用空板块]
Stage 2  ROI 裁剪 + 放大（≥512px）
         ├─ 手 crop = YOLO手框 ∪ 腕部关键点框（并集兜底）
         ├─ 脚 crop = 踝+脚趾关键点外扩框    [新增]
         ├─ 脸 crop = YOLO脸框
         └─ 躯干 crop = 肩-髋关键点框        [特殊部位结构判定入口]
Stage 3  几何规则（毫秒级，先筛）
         ├─ 现有 body/hand 规则             [保留]
         ├─ 新增脚趾规则（趾序/趾长/脚跟比例）
         └─ 可选：hand crop 上重估 21 关键点（RTMPose-hand ONNX）再跑规则
Stage 4  VLM 分区判定（核心新增）
         输入 = 全图(≤1024) + 各 ROI crop（多图一次请求，消融模型）
         输出 = 严格紧凑 JSON（≤250 token）：每区域 ok/deformed/not_visible + 问题列表
         解析失败 → 重试一次 → N/A+flag（绝不静默放行）
         按 (图像hash, prompt版本, 模型版本) 磁盘缓存
Stage 5  角色还原（三层：组内自洽 + 变体参考 + VLM 语义核对，见 §6）
Stage 6  融合
         硬门禁：重度解剖畸形 → 直接否决
         加权：structure + vlm_judge + character + quality + aesthetic
```

**级联模式（默认）**：全量跑 Stage 1-3（毫秒级）；VLM 只判 ① 几何 N/A/可疑图 ② 每组粗排 top-k → 调用量降至 20-30%。可切全量模式。

---

## 4. SFW 技术框架（表述约束，非防拒答机制）

- **助手侧**：本项目的调查、方案、代码、提示词全程以"动漫插画生产线结构质检"的纯技术框架表述，只讨论解剖结构/比例/连接/对称性等工程问题，不做内容题材分类。
- **VLM 侧**：默认消融模型（用户已确认路线），提示词用工业质检角色 + 技术措辞 + 严格 JSON。crop 策略的主要目的是**判定精度**（放大细节、排除无关上下文），顺带最小化输入上下文。
- **兜底**：JSON 解析失败 → 重试一次 → 该区域 `not_visible` + flag 进反馈日志。失败永远不等于"通过"。
- 采样：`temperature=1.0, top_k=20, top_p=0.95`（Qwen3 官方默认；忌 temp=0 复读）。

---

## 5. 解剖合理性判定最大化（优先级 ①）

每个部位至少两个独立信号源，几何 N/A 的盲区由 VLM 补：

| 部位 | 几何/检测信号 | VLM 信号（crop 放大后） |
|---|---|---|
| 肢体总数 | 骨架人数/肩髋结构 | 全图：多臂/多腿/缺肢 |
| 关节融合/不对称 | 现有规则（双肩/双腕融合、左右肢长比>1.35、零长节段） | 躯干 crop：连接是否自然 |
| **手指** | 现有 4 条规则 + crop 上可选关键点重估 | **手 crop：指数≠5、并指、多指、缺指、反关节**（核心修复） |
| **脚趾** | **新增**：DWPose 17-22 关键点，趾序/趾长比/脚跟-趾距 | **脚 crop：趾数、形态、融合** |
| 特殊部位 | 无（几何不适用） | 躯干 crop：比例/对称/连接结构合理性（纯结构判定） |
| 脸 | YOLO 脸框置信度 | 脸 crop：五官结构错位/融合 |

VLM 输出 schema（紧凑，控 token）：
```json
{"regions":[{"r":"hand_l","v":"deformed","i":[{"t":"extra_finger","s":0.9}]}]}
```
聚合：`score = 1 − Σ(severity × 区域权重)`；`severity ≥ 0.8` → 硬 flag（Stage 6 否决候选）。issue 原文进反馈 JSONL 供校准。

**为什么 crop 是关键**：现在手部判定发生在 288×384 全身图里的 ~30px 手上；crop 放大到 512px 后判定的是干净放大的手——ADetailer 修手的同一原理（"100px 的手放大到 512 再处理效果 dramatically better"），是"手指识别不准"最直接的物理解法。

---

## 6. 角色还原度：分组感知的三层设计（优先级 ②）

**用户场景**：每批图分若干组，仅 2-3 组用默认服装，其余组换服装甚至换发型。
**CCIP 事实**：发型是强敏感特征（换发型会推高差异值）；换装鲁棒性较好（训练数据含服装变体，merged_v1 明确合并泳装/职阶变体）；发色/肤色弱。
→ 不硬扛单一阈值，用三层互补：

### Layer A：组内自洽（零配置，主力）
同一组所有图共享同一 prompt 意图（同角色/同服装/同发型）→ 组内逐人 crop 提 CCIP 特征 → 质心/medoid → **离群图 flag**（diff > 阈值）。
- 不需要任何参考图，**任意服装/发型都成立**——组自己定义目标外观。
- 抓"组内漂移"：个别图画崩、画成别人、LoRA 不稳定。
- 成本：每图一次特征提取 + 组内差异矩阵（`ccip_batch_differences`），毫秒级。
- 盲区：整组全错（LoRA 根本没触发）→ 靠 B/C。

### Layer B：绝对身份（变体感知参考图）
`characters.yaml` 的 outfits 结构已存在，扩展为：
```yaml
- id: velina
  name_tags: ["velina airgid"]
  lora_patterns: ["Char-ZZZ-Velina"]
  invariant_features: "yellow eyes, ginkgo earrings"   # 跨服装不变量，供 VLM 核对
  outfits:
    - name: default
      reference_images: ["refs/velina/default/01.png", "..."]   # 3-10 张
    - name: swimsuit
      reference_images: []    # 可空 → 动态补充
```
- 默认服装组 → 默认参考图，阈值 0.213（b36 官方 F1 最优点）。
- 换装组：有变体参考图 → 用变体参考；没有 → **动态锚定**：该组第一张人工认可（feedback 点赞）的图自动成为组参考，后续图与之比对。零预配置成本。
- **换发型组**：CCIP 阈值放宽（1.3-1.5×，按组 prompt 发型 tag 与角色默认发型差异自动判定）+ 降权，身份判定交给 Layer C 的不变量核对。
- 无参考图且组内样本不足 → N/A（不瞎判）。

### Layer C：VLM 语义核对（补 CCIP 盲区 + 回答"还原度"本义）
- 把**该组 prompt 的服装/发型 tags** 作为 checklist，VLM 逐项核对"图中是否呈现了要求的服装件/发型"——这直接回答"还原度 = 是否画出了这组要求的样子"，而不只是"是否同一角色"。
- 核对跨服装不变量（`invariant_features`：瞳色/发色/标志配饰）——正好补 CCIP 对颜色弱的短板。
- 整组失败（LoRA 没触发、画成别人）只有 B/C 能抓。
- 与 Stage 4 合并为一次 VLM 请求（同图多任务），不额外增加调用。

**判定合成**：character_score = f(A 离群度, B 参考差异, C checklist 命中率)；严重不符 → flag。三层信号全部进反馈 JSONL，P3 校准权重。

---

## 7. 美观度（优先级 ③）

- 现有 aesthetic（shadow-v2 等 4 后端）**不动**。
- 解剖/还原上线校准后再重调权重；硬门禁保证美观分高埋不掉畸形。
- 顺序刻意如此：感知层及格之前调美观权重没有意义（这正是之前调权重无效的原因）。

---

## 8. 部署、显存与吞吐加速

### 8.1 显存预算（RTX 5090 Laptop 24GB）

| 组件 | 占用 |
|---|---|
| 便携 llama-server + Qwen3-VL-8B-abliterated Q4_K_M + mmproj-f16 + 8K ctx（Vulkan） | ~6-7GB（实测可启动） |
| DWPose(HumanArt/anime det + dw-ll) + ADetailer YOLO×2 + CCIP（全 ONNX） | ~2GB |
| aesthetic shadow-v2（1.1B ViT） | ~2.5GB |
| **合计** | **~16-17GB，留有余量** |

llama-server 参数：`--flash-attn on`、`--parallel 2`、`-c 8192`、`-ngl 99`（vlm_server 默认）；ctx/并发由 curator 配置传入，无 Modelfile 覆盖问题。
curator 侧 VLM 走 **OpenAI 兼容 HTTP**（`/v1/chat/completions`，images base64）→ 不引入 torch 依赖；任何既有 OpenAI 兼容端点可经 `vlm_judge.base_url` 复用（此时不 spawn 子进程）。

### 8.2 吞吐加速（5-10s/图 → 目标有效 0.3-1s/图）

原估算口径：q8_0 单流 ~40-60 tok/s × 400-600 输出 token。加速杠杆（按收益排序）：

| 杠杆 | 做法 | 预期收益 |
|---|---|---|
| **级联门控** | VLM 只判可疑图 + 每组 top-k（默认 20-30% 调用率） | 有效成本 ×0.2-0.3 |
| **输出压缩** | 严格紧凑 JSON（§5 schema），禁解释/禁 thinking，max_tokens ~250 | ×2-2.5 |
| **并发** | `llama-server --parallel 2-4`（vlm_server 默认 2）；8K ctx ×4 slot KV 仅 ~2-3GB | 聚合 ×2-3 |
| **量化档位** | 默认 Q4_K_M（5.03GB）；Q8_0 作精度 A/B 档，评测集验证后定 | ×1.3-1.5 |
| **小模型判 crop** | 手/脚 crop 是简单计数任务 → 4B（甚至 2B）判 crop，8B 只判躯干/全图 | crop 部分 ×~2 |
| **视觉 token 控制** | crop ≤512px、全图 ≤1024px、每请求 ≤4 crop | 编码耗时下降 |
| ~~远程批处理~~ | **已否决**（用户要求本地自给自足、可随仓库分发）：不依赖任何远端推理 | — |
| **长期：专用分类器** | 反馈数据积累后微调小型 ViT 手/脚畸形分类器（ms 级），替代多数 VLM crop 调用 | crop 判定 ×100+ |

**组合估算（待真实图库基准，非承诺）**：本地 Q4_K_M + 紧凑 JSON + 并发 2-3 → 全量 VLM ~1-2s/图；级联后有效 ~0.3-0.6s/图，**1000 张 ≈ 5-10 分钟**。缓存命中后重复扫描零成本（实测：冷调用 18.2s 含首载，缓存命中 0.3s）。
**纪律**：P1 先 warmup 再实测 tok/s 与端到端延迟（含视觉编码），用评测集比较 q8/q4/4B 的精度-速度曲线后定档；不凭估算定配置。

---

## 9. 实施阶段

| 阶段 | 内容 | 预估 |
|---|---|---|
| **P0** | 检测器 A/B（HumanArt YOLOX vs deepghs anime YOLOv8，100 张实测）→ 胜者替换；YOLO 板块填 hand_yolov9c + face_yolov8n_v2；脚趾几何规则；评测集骨架（100-200 张标注：手/趾/肢体/特殊部位/角色不符/正常） | 0.5-1 天 |
| **P1** | `vlm_judge.py` 真实现：便携 llama-server 后端（零安装）+ ROI crop + 紧凑 JSON + 解析兜底 + 磁盘缓存 + **shadow 模式**；吞吐基准（q4/q8/4B × 并发档位）定档 —— **代码已落地，待真实图库基准** | 1-2 天 |
| **P2** | CCIP 三层角色 scorer（组内自洽 + 变体参考/动态锚定 + VLM checklist）+ 配置扩展 | 1 天 |
| **P3** | 反馈校准（离线 replay → shadow 转正 → `docs/CALIBRATION_*.md`）：硬门禁阈值、区域权重、composite 权重；专用手/脚分类器视数据量启动 | 持续 |

遵守仓库既有纪律（docs/CALIBRATION_2026-10.md）：**新信号先 shadow（记录+展示+不进综合分），反馈离线 replay 验证 top-1 命中率与问题图召回后再转正**；绝不凭直觉调权重。

## 10. 验收标准

- 标注评测集：手指/脚趾畸形召回 ≥80%（相对现有 structure 显著提升），误报率 ≤15%；
- 特殊部位畸形：从 0 检出 → 可检出；
- 角色不符：换装/换发型组均可判（组内自洽 + 变体参考 + checklist），整组失败可抓；
- 全量扫描吞吐与显存符合 §8（**以实测为准**）；VLM 输出解析失败率 <1% 且全部有 flag。

---

## 附：证据来源

- Qwen3-VL-8B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- 便携运行时：llama.cpp releases https://github.com/ggml-org/llama.cpp/releases （b12322，Windows Vulkan 构建；零安装，`.models/` 内自包含）
- 消融版 GGUF（默认权重）：https://huggingface.co/mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF ；原版消融 https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated
- rtmlib 官方模型表（YOLOX COCO vs HumanArt AP、RTMDet-nano hand AP 76.0、"HumanArt 可检测真人+卡通"注记）：https://github.com/Tau-J/rtmlib
- HumanArt 数据集（50k 图/20 场景/COCO 联合训练）：https://github.com/IDEA-Research/HumanArt （CVPR 2023, arXiv 2303.02760）
- deepghs 动漫人体检测（F1 表/训练集构成）：https://hub.deepghs.org/models/deepghs/anime_person_detection ；https://huggingface.co/datasets/deepghs/anime_person_detection ；API https://dghs-imgutils.deepghs.org/main/api_doc/detect/person.html
- ADetailer 模型库：https://huggingface.co/Bingsu/adetailer ；动漫手替代 https://github.com/styler00dollar/YOLO-anime-hands
- CCIP 模型卡与基准：https://huggingface.co/deepghs/ccip ；API https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html ；局限 https://deepghs.org/waifuc/main/advanced_guides/ccip/index.html ；merged_v1 变体合并 https://hub.deepghs.org/datasets/deepghs/character_similarity
- 远程 27B 无 vision：2026-11 实测 `http://ai1.cafegame.cn:18088/api/show` → capabilities [tools, thinking, completion]
- 显存/KV：https://netray.co/on-prem-ai/models/qwen3-vl-8b-instruct 、https://jacar.es/en/local-multimodal-models-16gb-vram/
