# react-color-agent

`react-color-agent` 是一个面向科研调色任务的轻量单 Agent ReAct 系统。系统由一个
`Experiment Director` 驱动，确定性能力以工具和 Skill 的形式提供给 Agent：Agent 负责
读取在线证据、解释实验结果、选择分析方法、设计下一批实验并说明理由；工具负责材料查询、
配方与体积校验、光谱解析、CIE 计算、数据索引、XLSX 生成和状态持久化。

本项目的目标是跑通可恢复的科研闭环，而不是提供生产级实验室调度或多 Agent 平台：

```text
用户需求
  → PubChem 材料身份确认
  → Agent 读取 Skill 并设计 B1
  → 生成 XLSX 与空数据回传目录
  → 人工实验并回传光谱
  → 光谱 QC / CIE 计算 / 结果分析
  → 研究数据集与索引更新
  → Agent 按需读取历史证据并设计下一批
  → 确定性评价 → Scientific Critic → Agent 最终提交
  → 达标复测、继续迭代、停止或提交人工审核
```

## 快速开始

### 安装

项目要求 Python 3.11 或更高版本。开发环境和 OpenAI 客户端可以一起安装：

```bash
python -m pip install -e '.[dev,openai]'
```

其中 `openai` 是实际运行 Director 所需的可选依赖；不安装它仍可创建、查看任务，
也可以运行离线测试。

运行测试：

```bash
python -m pytest -q
```

如果没有安装命令行入口，也可以从仓库根目录运行：

```bash
PYTHONPATH=src python -m react_color_agent.cli --help
```

### 配置 `.env`

CLI 每次启动时从**当前工作目录**读取 `.env`。Shell 中已经存在的同名变量优先于文件值。
常用配置如下：

```dotenv
OPENAI_API_KEY=your-api-key
BASE_URL=https://your-openai-compatible-endpoint/v1
BASE_MODEL=your-director-model
CRITIC_MODEL=your-critic-model
```

`BASE_URL` 会自动映射为 OpenAI SDK 使用的 `OPENAI_BASE_URL`；也可以直接配置
`OPENAI_BASE_URL`。Director 使用 `BASE_MODEL`，Scientific Critic 使用 `CRITIC_MODEL`；
未配置时 Critic 回退到 Director 模型，Director 默认模型为 `gpt-4.1-mini`。

### 创建任务

以下是完整示例：

```bash
color-agent create \
  --task-id cie-0345 \
  --material '罗丹明6G 罗丹明B 2-溴-9,10-二苯基蒽 香豆素6 9,10-二苯基蒽' \
  --target 0.345,0.345 \
  --max-rounds 3
```

单次 `--material` 参数会按空白拆分，适合中文材料名批量输入。材料名称本身含空格时，
重复参数并将每个完整名称作为一个 Shell 参数传入：

```bash
color-agent create --task-id demo \
  --material 'Rhodamine 6G' --material 'Coumarin 6' \
  --target 0.345,0.345
```

实际材料数量可以是 1–5 种。为了兼容 CPWL 模板和历史数据，配方始终使用固定的五个槽位：

| 槽位 | 含义 |
| --- | --- |
| A | 第 1 种材料 |
| B | 第 2 种材料 |
| C | 第 3 种材料 |
| D | 第 4 种材料 |
| E | 第 5 种材料 |

未使用槽位必须填写 `0`。例如三种材料的配方仍是
`[0.00002, 0.00003, 0.00004, 0, 0]`，而不是三维数组。

## 启动与恢复

创建任务后启动 Agent：

```bash
color-agent run --run-dir runs/cie-0345 2>&1 | tee run.log
```

运行时会在终端显示 Director 的状态、工具选择、工具结果摘要、Critic 调用和暂停原因；
标准输出保留最终状态 JSON，工具追踪默认输出到标准错误，因此可以使用上面的命令同时查看和保存日志。

首次运行通常会完成材料查询、Skill 读取、B1 设计，然后暂停在 `WAITING_FOR_DATA`。
如果外部 API 或决策预算耗尽，状态会被保存，之后直接重复同一命令即可恢复，不需要重建任务。

查看当前状态：

```bash
color-agent show --run-dir runs/cie-0345
```

`--max-rounds` 控制一次任务最多允许完成多少个实验批次；完成该数量后状态变为 `STOPPED`。
它不会限制单批配方数，也不会绕过当前批次的人工数据等待。

## Agent、Skill 与工具

当前系统是“一个 Agent + 确定性工具/Skill”的结构。Agent 通过工具调用完成工作，而不是
由固定 Python 流程替它决定所有实验路径。

### Agent 能力

- 将中文材料名转换为英文查询候选（转换说明写在 Prompt/工具描述中）。
- 调用 PubChem 确认 CID、名称、分子量和可用性质。
- 仅在 PubChem 身份已确认但信息不足时调用固定 Crossref REST 查询。
- 读取 CPWL Skill，设计 B1 或后续 2–12 组配方并给出可证伪的实验理由。
- 按需读取研究摘要、批次记录、具体样品和光谱，而不是每轮完整注入全部历史。
- 自主选择诊断、特征提取、局部模型、加权邻域、模型比较和候选生成工具。

### Skill

Skill 位于 `src/skills/`，由 `read_skill` 工具加载并写入任务快照。当前主要包括：

- `cpwl/experiment_plan_xlsx`：CPWL 配方、体积、XLSX 和说明文件合同。
- `cpwl/measurement_data_return`：人工回传目录与发射/吸收光谱文件合同。
- `cpwl/research_iteration`：读取历史证据、选择分析方法和设计后续实验的科研流程。

`experiment_design` Skill 已移除；实验设计由 Director 结合现有 Skill 和实测证据完成。

### 重要工具

| 类别 | 工具示例 | 作用 |
| --- | --- | --- |
| 在线证据 | `query_pubchem`, `search_crossref` | 查询材料身份与公开性质，不用本地材料数据替代在线证据 |
| 实验计划 | `design_initial_batch`, `design_followup_batch`, `propose_followup_batch`, `finalize_followup_batch` | 生成、评价并提交实验批次 |
| 光谱处理 | `ingest_spectra`, `calculate_cie`, `extract_spectral_features` | 校验并归档谱图，计算 CIE，提取峰值/面积等特征 |
| 研究数据 | `update_research_dataset`, `query_research_index`, `get_experiment_record`, `get_spectrum_data` | 保存并按需检索历史事实 |
| 分析建模 | `diagnose_dataset`, `screen_composition_effects`, `compile_research_analysis`, `fit_local_response_model`, `compare_models` | 让 Agent 比较数据处理方式并选择下一步 |
| 目标决策 | `check_goal`, `generate_predicted_candidates`, `propose_unreachable_request` | 判定目标、提出候选或申请人工审核 |

## CPWL 实验方案合同

### 母液、体积和浓度

以下约束来自 `src/constraints/constraint.md` 和
`src/skills/cpwl/experiment_plan_xlsx/cpwl_xlsx.py`，是确定性校验，不由 Agent 修改：

| 项目 | 固定值 |
| --- | ---: |
| 每组产物体积 | `8 ml` |
| A–E 母液浓度 | `0.0001 mmol/ml` |
| 单种材料终浓度上限 | `0.00006 mmol/ml` |
| 总终浓度范围 | `0.00004–0.0001 mmol/ml` |
| 任一母液或溶剂体积上限 | `4.8 ml` |

对每个配方，工具使用：

```text
V_i = c_i_final × 8 / 0.0001
V_solvent = 8 − ΣV_i
mg/ml = mmol/ml × 分子量(g/mol)
```

每个材料都必须提供分子量、`mmol/ml`、`mg/ml` 和对应母液体积；不能凭空补造缺失的
分子量、母液浓度或混合基准。XLSX 只包含合成行 `B{batch}-N{number}-S`，并生成与之
对应的说明文件。

### 批次规模与设备参数

- B1 最多 24 组配方。
- B2 及以后普通后续批次最多 12 组，通常由 Agent 提交 2–12 组。
- 后续批次保留至少一个有实测依据的竞争候选；不额外强制纯验证样品。
- 固定激发协议为 `350 nm`，新发射谱范围为 `360–760 nm`，步长为 `0.2 nm`。
- XLSX 中的吸收扫描范围仍为 `400–700 nm`，吸收文件本身按下述回传合同执行。

每个批准批次会生成：

```text
artifacts/experiment_plans/batch_001.xlsx
artifacts/experiment_plans/batch_001.md
artifacts/measurement_returns/round-001/
├── B1-N1-D/
├── B1-N2-D/
└── ...
```

## 人工数据回传

实验人员只需把文件放入系统生成的目录，然后用 `--data` 恢复；不要手工提交 CIE JSON。
检测目录名称严格使用 `B{batch}-N{number}-D`，其中 `D` 表示人工检测结果，不是 XLSX
中的检测行。

```bash
color-agent run \
  --run-dir runs/cie-0345 \
  --data runs/cie-0345/artifacts/measurement_returns/round-001 \
  2>&1 | tee round-001.log
```

### 发射光谱（必需）

`emission.txt` 为 UTF-8 文本，必须包含以下五行头部，之后是 2001 行制表符分隔数据：

```text
Scan Mode: 发射扫描
激发波长: 350 nm
发射波长范围: 360 - 760 nm
步长: 0.2 nm
波长(nm)\t荧光强度
```

波长必须严格为 `360.0, 360.2, …, 760.0`，升序且不重复；强度必须是有限、非负且总和大于零。
系统不会自动插值、排序、平滑、基线校正或修复新回传值。历史运行中的 `400–700 nm` 网格
仅用于读取旧数据，新实验必须使用当前网格。

### 吸收光谱（可选）

`absorption.txt` 不是 CIE 计算的硬性输入。若提供，文件头为：

```text
Wavelength(nm)\tTransmittance(%)\tAbsorbance
```

数据为 `700, 699, …, 400` 的 301 行整数波长。仪器产生的 `NaN/Inf` 吸光度会保留在原始
证据中，但该吸收谱会被排除出派生分析；缺失或无效吸收谱不阻塞发射谱 CIE 计算。有效吸收谱
可用于可选 QC、最大吸光度和后续光学机理分析。

系统使用 CIE 1931 2° 标准观察者对合格发射谱积分得到 `x,y`；CIE 由
`calculate_cie` 确定性计算，Agent 不应在自然语言中自行填写或修改 CIE。

## 自迭代闭环

每个批次完成后，系统把事实和解释分开保存：

- `artifacts/research_dataset.json`：材料、配方、原始测量引用、CIE、目标判定和批次记录。
- `artifacts/research_index.json`：按 CIE 距离、样品 ID、材料槽位和批次建立的轻量索引。
- `artifacts/research_notebook.md`：便于人工阅读的摘要。
- `artifacts/round-*/`：原始谱图、谱图清单、CIE、分析结果和设计结果审查。
- `state.json` 与 `snapshots/`：可恢复状态及状态更新前后的快照。

当目标未达成，Director 通常按以下顺序工作：

```text
读取紧凑研究简报
  → 查询目标附近样品与当前批次结果
  → review_design_outcomes
  → 选择问题驱动的诊断/建模方法
  → 形成 2–12 组草案
  → 确定性批次评价
  → 单次 Scientific Critic 审查
  → Director 逐条回应并 finalize_followup_batch
```

Scientific Critic 只负责指出配方、证据、光学机理和批次策略风险，不直接提交实验，也不
拥有停止权限。Director 仍是最终决策者。批次草案、确定性评价、Critic 意见和 Director
最终回应都会写入 artifacts，便于论文复核。

### 目标命中与复测

目标判定使用逐坐标容差：

```text
|x − x_target| ≤ 0.005 且 |y − y_target| ≤ 0.005
```

第一次达到容差只记录为临时候选；系统会要求同配方独立制备复测。初测与复测都达标后才进入
`FINISHED`。如果未达标且尚有轮次，则进入下一轮设计；达到 `max_rounds` 则进入 `STOPPED`。

### 不可达申请

当真实数据、已探索路线、实验约束和剩余预算支持“当前边界内继续实验暂不值得”的判断时，
Agent 可以调用 `propose_unreachable_request`。申请必须引用真实实测证据，不能使用
`synthetic_dry_run`，并先由 Unreachable Scientific Critic 审查；Director 复核后才能提交人工审核。

```bash
color-agent review-unreachable \
  --run-dir runs/cie-0345 \
  --decision approve \
  --reason '已核对实测证据，当前探索范围内不再继续。'
```

批准后任务才会停止；拒绝则保留人工理由并回到 `DESIGNING`。不可达表示当前任务边界下不建议
继续，并不表示普遍的物理或化学不可能。

## 运行限制与防循环

一次 Director 决策最多执行 40 个工具步骤。为防止 Agent 在检索阶段循环调用：

- 连续只读工具调用上限为 10。
- `query_research_index` 在一次设计阶段最多 5 次。
- 重复相同索引参数或重复得到同一结果会返回可恢复的拒绝观察，要求 Director 改变行为。
- Scientific Critic 返回后，Director 必须调用最终提交工具，不能只输出自然语言。

预算耗尽时系统保存当前状态并暂停；后续重新执行 `color-agent run` 即可恢复。预算限制只限制
当前 Agent 决策过程，不会删除已经保存的实验数据、谱图或研究索引。

## 从已有批次派生新任务

可以保留一个已有 run 截止指定批次的真实事实，同时重新开始后续设计：

```bash
color-agent branch-run \
  --source-run runs/cie-0333-83 \
  --through-round 1 \
  --run-dir runs/cie-0333-83-review
```

目标目录必须不存在。命令会复制指定轮次的原始谱图、实测 CIE、配方和研究索引，不修改来源
run，也不会继承指定轮次之后的 Agent 决策或模型派生物。

## 论文证据采集

运行时的旁路采集文件为：

```text
evaluation/runtime_trace.jsonl
```

它记录状态观察、Agent 工具选择、工具耗时、结果摘要、可用的 prompt/completion/total token
使用量，以及当前源码、Prompt、Skill、模型、CIE 容差、光谱合同和预算的 protocol fingerprint。
该采集层不注册工具、不改变任务状态、不参与实验决策。

任务暂停、完成或停止后生成论文材料：

```bash
color-agent collect-evidence --run-dir runs/cie-0345
```

输出位于 `artifacts/paper_evidence/`，包括运行清单、批次指标、样品 CIE、原始谱图哈希、
预测与实测对照、决策 trace、按顺序配对的 `tool_trajectory.json` 和 `paper_summary.md`。
`synthetic_dry_run` 只保留操作 lineage，不进入科学指标、模型或论文结论。

## 任务目录

一个典型任务目录如下：

```text
runs/cie-0345/
├── state.json
├── snapshots/
├── artifacts/
│   ├── evidence/
│   ├── skills/
│   ├── experiment_plans/
│   ├── measurement_returns/
│   ├── round-001/
│   ├── research_dataset.json
│   ├── research_index.json
│   ├── research_notebook.md
│   └── paper_evidence/
└── evaluation/
    └── runtime_trace.jsonl
```

`state.json` 是当前可恢复状态，`research_dataset.json` 是跨轮次科学事实的主记录，原始谱图
和每次工具输出则保留在对应 artifact 中。不要手工修改这些文件来伪造测量结果；如需从历史
批次重新开始，使用 `branch-run`。

## 开发结构

所有项目资源位于 `src/`：

```text
src/react_color_agent/     # Agent、状态、存储、CLI、评估采集
src/react_color_agent/director/
src/skills/                 # Agent 可读取的科研 Skill
src/tools/                  # 普通确定性工具
src/constraints/            # 人工维护的 CPWL 约束与模板
```

修改固定实验约束前，应先同步更新 `src/constraints/constraint.md`、相关 Skill、验证逻辑和
本文档；不要让 Agent 的自迭代过程直接改写约束源。
