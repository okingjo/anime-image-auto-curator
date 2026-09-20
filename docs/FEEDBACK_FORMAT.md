# 反馈数据采集格式（FEEDBACK_FORMAT.md）

> 本文件是给「以后的分析者」（人或 AI）看的正式规格。改动格式时必须升级 `schema` 版本号并更新本文档。

## 用途

用户每次在 WebUI 勾选「记录本次导出」并导出（JSON 或拷贝）时，系统把**本次会话的完整筛选上下文**追加为一行 JSON，写入：

```
data/feedback/feedback-YYYY-MM-DD.jsonl    （按天一个文件，UTF-8，每行一条 JSON）
```

积累一段时间后，用这些数据分析：

1. **哪些结构规则 flag 与"用户不选/换选"相关** → 校准阈值；
2. **structure 影子分与人工选择的秩相关（Spearman）** → 决定何时退出影子模式、给多少权重；
3. 特征充分后可直接训练小分类器（LightGBM 等）替代手工规则（Phase C）。

> `data/` 已加入 .gitignore —— 反馈含个人图片路径与 prompt，**不进仓库**。

## 记录时机

- 仅在导出时记录，且仅当用户勾选了「记录本次导出」。
- 一次导出 = 一行 JSONL = 一次完整会话快照（含所有组、所有图、所有分）。
- 未勾选导出不产生任何记录。

## 顶层结构（schema: `aic.feedback.v1`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema` | str | 固定 `"aic.feedback.v1"`，格式变更必须升级 |
| `session_id` | str | 本次进程会话的 12 位随机 id（同一会话多次导出可关联） |
| `ts` | str | ISO-8601 带时区，导出时刻 |
| `mode` | str | `"json"` \| `"copy"`，本次导出方式 |
| `root` | str | 扫描的图片根目录 |
| `weights` | dict | 当时的综合分权重（quality/aesthetic/...） |
| `aesthetic_backend` | str\|null | 当时激活的美观度模型 id（shadow-v2 等） |
| `structure_cfg` | dict\|null | 当时的 structure 完整配置（shadow/parts/pose/yolo，含各 YOLO 模型的 source） |
| `scorers` | list | 各打分器 `{name, available, shadow}` |
| `n_groups` | int | 组数 |
| `exported` | list\|null | 本次导出的文件（copy=目标路径列表；json=文件名列表） |
| `groups` | list | 每组一条，见下 |

## group 条目

| 字段 | 类型 | 说明 |
|---|---|---|
| `key` | str | 组键（规范化后的 positive‖negative prompt） |
| `character_id` | str\|null | 识别到的角色 |
| `positive` | str | 正面 prompt（截断到 2000 字符） |
| `recommended` | str\|null | 系统推荐（综合分最高）的文件名 |
| `selected` | str\|null | **用户最终选择**；`null` = 用户点了「全部不选」（这是重要信号：整组质量都不行） |
| `members` | list | 组内每张图的完整记录，见下 |

## member 条目（每张图）

| 字段 | 类型 | 说明 |
|---|---|---|
| `filename` / `path` | str | 文件名 / 绝对路径 |
| `width` / `height` | int | 像素尺寸 |
| `positive` / `negative` | str | 该图的完整 prompt |
| `seed` / `model` | str\|null | 生图种子 / 底模（来自图内嵌参数） |
| `loras` | list[[name, weight]] | 使用的 LoRA |
| `character_id` / `outfit` / `outfit_ratio` / `outfit_missing` | — | 角色与服装匹配 |
| `composite` | float | 综合分（0–1，影子项不参与） |
| `scores` | dict | 每个打分器的完整结果，见下 |

### scores.<scorer> 通用结构

```json
{ "score": 0.0-1.0, "subs": {...}, "flags": ["..."], "notes": "..." }
```

- `flags`：人类可读的问题标记（中文），如 `"左手指尖聚集(并指嫌疑)"`。
- 打分器缺席（不可用/N/A）时该 key 不存在。

### scores.structure 特有字段（分析重点）

```json
{
  "score": 0.83,
  "flags": ["左手手指长度顺序异常(小指>中指)", "手低置信(0.42)"],
  "notes": "shadow",
  "subs": {
    "overall": 0.83,
    "body": 0.85,          // 骨架子分，null=N/A（人体都不可见）
    "hands": 0.80,         // 手子分（各可见手的均值），null=N/A（手不可见/太小）
    "yolo": 0.55,          // YOLO 板块分（各模型分的平均），null=N/A（全部模型无检出）
    "visible": {           // 可见性清单
      "person": true, "hand_l": true, "hand_r": false, "feet": true,
      "yolo_parts": ["hand"]      // 有检出的 YOLO 模型 id
    },
    "feats": {             // ★ 完整特征行，供离线分析/训练
      "rules_version": "v1",
      "n_persons": 1,
      "pose_error": "",
      "parts": {"body": 0.85, "hands": 0.80, "yolo": 0.55},
      "body": [            // 每个人一条
        {
          "torso_px": 412.0, "shoulder_w_px": 180.5, "neck_ratio": 0.62,
          "body_conf_mean": 0.81, "feet_conf": 0.33,
          "手臂_asym": 1.08, "腿_asym": 1.02,   // 左右肢长比（>1.35 触发 flag）
          "body_score": 0.85
        }
      ],
      "hands": [           // 每只手一条
        {
          "side": "l", "visible": true, "too_small": false,
          "conf_mean": 0.72, "palm_px": 55.3,
          "tip_dists": {"食指": 96.1, "中指": 104.2, "无名指": 99.8, "小指": 88.0},
          "bad_segments": 0, "tip_clusters": 1,
          "hand_score": 0.75
        },
        {"side": "r", "visible": false, "conf_mean": 0.11}   // 不可见 → 不参与
      ],
      "yolo_models": [     // 每个配置的 YOLO 模型一条
        {"id": "hand", "label": "手", "available": true, "detected": true,
         "n_det": 2, "max_conf": 0.91, "score": 0.55},
        {"id": "face", "label": "脸", "available": true, "detected": false}
      ]
    }
  }
}
```

## 分析建议（给未来的自己/AI）

1. **样本定义**：一条 member 记录 = 一个样本；标签 = `selected == filename`（选中=1），
   `selected == null` 的组内全部 member 标签为 0（整组弃选）。
2. **同组对照优先**：同组内被选 vs 未被选是天然配对样本，控制了 prompt/底模变量，比跨组比较干净得多。
3. **N/A 语义**：`body/hands/yolo == null` 表示"该部分不可见"，分析时单独归类（不可见 ≠ 有问题）。
   `visible` 字段用于区分"检出了但分低"（真问题）vs"没检出"（N/A）。
4. **阈值校准**：对每个 flag，统计其在 被选/未选/弃选组 中的出现率差异；差异小的规则降权或删除。
5. **退出影子模式的条件**：structure 影子分与人工选择在 ≥300 个配对样本上 Spearman ρ 稳定为正，
   且弃选组的 structure 分显著低于被选组（Mann-Whitney p<0.05），再考虑给正式权重（起步 0.2）。
6. **特征行**：`subs.feats` 每行可直接展开为训练特征（配合 rules_version 过滤版本漂移）。

## 历史版本

- `aic.feedback.v1`（2026-09）：初版。structure 影子分 + DWPose 特征 + YOLO 板块。
