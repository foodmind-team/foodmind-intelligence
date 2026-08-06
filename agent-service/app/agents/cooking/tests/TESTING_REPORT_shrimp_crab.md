# 基围虾 + 蟹脚 双菜谱 E2E 测试报告

> 执行日期：2026-08-06 | 方式：直连 workflow graph（`build_cooking_plan_graph`）
> 复现脚本：[scripts/test_shrimp_crab.py](../scripts/test_shrimp_crab.py)
> 运行：`uv run python scripts/test_shrimp_crab.py`

---

## 1. 测试输入

用户提交两道菜谱原文（各 4 人份），经规则解析器（LLM 关闭）抽取后进入完整 workflow：
解析 → gap 检测/本地推理 → 可行性检查 → 排程（CP-SAT）→ 独立验证 → 渲染。

| 菜谱 | recipe_id | 抽取食材 | 抽取步骤 | 文本份数 |
|---|---|---|---|---|
| 基围虾（豆瓣酱香辣） | `r_shrimp` | 12 项 | 8 步 | 无（默认 2） |
| 蟹脚（香辣焖煮） | `r_crab` | 14 项 | 7 步 | 无（默认 2） |

> 注意：文本未声明份数，抽取默认 2 人份；请求 `target_servings=4` 会令 IR 需求翻倍。

---

## 2. 场景 A：常规库存 → `READY`（修复解析层缺陷后）

用覆盖全部常见食材的常规 mock 库存即可跑出完整排程。

**修复前的表现**：`NEEDS_CONFIRMATION`，4 个修复选项集中在两处无法匹配库存的食材——
`味精/鸡精（可选）` 与 `白胡椒粉、`（名称污染导致）。

**修复后的表现**：`READY`，所有食材均能匹配常规库存，无需确认。

| 指标 | 值 |
|---|---|
| 状态 | READY |
| 求解器 | OPTIMAL |
| 验证器 | passed=True，issues=0 |
| 总耗时（makespan） | 85 分钟 |
| 时间线任务 | 22 个 |
| recipe_tasks | 19 个 |

---

## 3. 场景 B：库存充足 → `READY`（完整排程）

按 parse→IR 需求口径自动补足库存（仅测试用），与场景 A 结果一致。

**关键指标**

| 指标 | 值 |
|---|---|
| 状态 | READY |
| 求解器 | OPTIMAL |
| 验证器 | passed=True，issues=0 |
| 总耗时（makespan） | 85 分钟 |
| 时间线任务 | 22 个 |
| recipe_tasks | 19 个 |

**各菜完成时间**

| 菜品 | 完成分钟 |
|---|---|
| shared（共享备料） | 14 |
| r_crab（蟹脚） | 80 |
| r_shrimp（基围虾） | 85 |

**排程亮点（并行复用）**：蟹脚焯水 [26–36 分钟 Heat] 与虾的处理/切配 [26–31、31–36] 并行；
蟹脚焖煮 [49–55 分钟 Heat] 期间插入虾的小米辣切配 [49–54] 与煎虾 [54–59] —— 被动等待期被充分填充。

**Mise en place（3 项）**：一次取出所有食材/调料/工具（5 分钟）、集中清洗沥干（8 分钟）、
确认清洗切配与调料分装完成（1 分钟）。

---

## 4. 解析层缺陷修复记录

修复位于 `src/cooking_plan_agent/parsing/`（`extractor.py` + `extractor_patterns.py`）。

| # | 输入原文 | 修复前 | 修复后 | 处理位置 |
|---|---|---|---|---|
| 1 | `白胡椒粉、` | name=`白胡椒粉、`（尾标点未清洗） | name=`白胡椒粉` | `_clean_ingredient_name` 剥离尾标点 |
| 2 | `味精/鸡精（可选）`、`干辣椒（可选）` | `（可选）`标记未剥离；无法匹配库存 | `味精/鸡精` 拆分为 `味精`+`鸡精`；`干辣椒` | `_RE_PAREN_NOTE` 剥离括号注释 + `_expand_slash_alternatives` 拆分 `/` 候选 |
| 3 | `老抽少许` | name=`老抽少许`（"少许"未解析） | name=`老抽` | `_RE_QUANTITY_QUALIFIER` 识别任意位置量词 + 尾量词剥离 |
| 4 | `大蒜3-4瓣` | qty=3, prep=`-4瓣`（范围拆分错误） | qty=4, unit=`piece` | 中文正则支持数量范围取上限 + `瓣`→`piece` |
| 5 | `小米辣（依吃辣程度放）` | **从食材列表完全丢失** | name=`小米辣` | 括号注释剥离后再做步骤指示词判定 |
| 6 | 无数量食材（蟹脚等） | 名称污染 → 无法匹配库存 | 名称干净 → 可匹配库存 | `_clean_ingredient_name` 统一清洗 |

新增单元测试（`tests/unit/parsing/test_parser_pipeline.py`）：
`test_extract_noise_free_names`、`test_extract_trailing_punctuation_cleaned`、
`test_extract_quantity_range_and_slash_alternatives`。

**回归验证**：parsing + normalisation 216 个测试通过；全量 `pytest tests/` 1070 通过。
contract/task 目录 12 个失败为既有 LangGraph checkpointer 配置问题（`thread_id` 缺失），
与本修复无关（stash 改动后同样失败）。

---

## 5. 复现方式

```bash
cd agent-service/app/agents/cooking
uv run python scripts/test_shrimp_crab.py
```

输出两个场景：场景 A（常规库存）与场景 B（自动补足），修复后均为 READY。
