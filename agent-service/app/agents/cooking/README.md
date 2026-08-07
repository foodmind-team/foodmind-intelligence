# FoodMind Cooking Plan Agent

把多菜谱请求转换为可执行的烹饪排程（时间线、mise en place、完成清单）的智能体服务。

- 输入：1–6 个菜谱的原始文本（或预解析候选），可选库存/厨房资源/饮食约束
- 输出：四种终态 —— `READY`（已验证排程）、`NEEDS_CONFIRMATION`（需用户确认）、`INFEASIBLE`（不可行）、`FAILED`（错误）
- 架构：LangGraph 16 节点工作流 + CP-SAT（OR-Tools）词典序多目标排程 + 独立验证器

## 技术栈

- Python 3.13 + `uv`、FastAPI、Pydantic v2（`src` 布局）
- LangGraph（工作流）、OR-Tools CP-SAT（排程）、SQLite（任务/checkpoint 持久化）
- Ruff（lint/format）、mypy strict、pytest + coverage

## 快速开始

```bash
uv sync
uv run uvicorn cooking_plan_agent.main:app --reload --host 127.0.0.1 --port 8003
```

健康检查：`GET /health/live`、`GET /health/ready`、`GET /health/load`（绕过背压限流器）。
API 文档：`GET /docs`。OpenAPI 契约导出：`uv run python scripts/export_openapi.py`。

> 直接 `uv run uvicorn` 前需确保 `PYTHONPATH=src`（或通过 `uv run --with` 安装 editable 包）。

## 关键配置（环境变量前缀 `COOKING_PLAN_`）

| 配置 | 默认 | 说明 |
|---|---|---|
| `INTERNAL_SERVICE_TOKEN` | —（必填） | Spring 侧调用鉴权 token（非 local 环境 ≥16 字符） |
| `SAFETY_POLICY_REGION` | `US` | 部署级默认食品安全政策地区；请求 `region` 字段可覆盖 |
| `SAFETY_POLICY_VERSION` | 无（取最新） | 显式策略版本；旧版本仅保留供历史审计 |
| `SOLVER_OPTIMIZATION_LEVEL` | `full` | 排程优化深度：`makespan` / `phase12` / `full`（词典序） |
| `SOLVER_TIMEOUT_SECONDS` | `5.0` | CP-SAT 求解超时 |
| `CHECKPOINT_ENABLED` / `CHECKPOINT_BACKEND` | `false` / `sqlite` | 工作流断点持久化（P2-06） |
| `TASK_API_ENABLED` | `false` | 异步任务 API（P3-01，进程内 worker） |
| `LLM_ENABLED` / `LLM_BASE_URL` / `LLM_MODEL` | `false` / … | LLM 抽取开关（默认走规则抽取） |
| `EXPLANATION_ENABLED` | `false` | READY 响应附带排程解释（P4-01）；关闭时 `explanation` 为 `null` |

## 多地区食品安全策略（P3-04）

安全阈值（蛋白中心温度、静置、热/冷 holding、复热）不再硬编码，而是来自可版本化、
有官方来源的地区策略包（`src/cooking_plan_agent/safety/policies/`）：

| 地区 | 版本 | 危险区 | 室温 holding | 禽肉中心温度 | 复热 |
|---|---|---|---|---|---|
| `US`（USDA FSIS） | 1.0 | 4–60°C | ≤2 小时 | 74°C | ≥74°C |
| `SG`（Singapore SFA） | 1.0 | 5–60°C | ≤4 小时（EPH 条例 13A） | 75°C | ≥75°C 且持续 ≥2 分钟 |

- 地区选择是**显式的**：请求 `region` 覆盖部署默认；未知地区、未知版本、未生效、
  缺来源的策略一律拒绝并返回 `SAFETY_POLICY_UNAVAILABLE`（FAILED），绝不静默回退。
- READY / NEEDS_CONFIRMATION 响应携带 `safety_policy`（region/version/sources）供溯源。
- 阈值只来自官方来源与代码审查（D7），LLM/普通搜索不能修改。

## 错误码（节选）

| 错误码 | 含义 | 处理 |
|---|---|---|
| `SAFETY_POLICY_UNAVAILABLE` | 地区策略未知/未生效/缺来源 | FAILED，客户端修正 region |
| `SCHEDULE_INFEASIBLE` | 求解器证明有效模型无解 | INFEASIBLE（业务终态） |
| `SCHEDULE_MODEL_INVALID` / `SCHEDULE_UNKNOWN` | 模型非法 / 求解超时未定 | FAILED |
| `SCHEDULE_VERIFICATION_FAILED` | 独立验证器拒绝求解结果 | FAILED（P1 告警） |
| `OVERLOADED` / `SHUTTING_DOWN` | 背压 503 / 停机 503 | 带 `Retry-After` |

完整目录见 `src/cooking_plan_agent/domain/errors.py`；运维步骤见 `runbook.md`。

## 质量门禁

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=cooking_plan_agent --cov-report=term-missing
uv run pytest tests/contract tests/security -v
uv run python scripts/export_openapi.py --check
```

## 文档

- 开发计划：`docs/development-plans/`（P0 正确性 → P1 韧性 → P2 能力 → P3 架构）
- 交付清单：`docs/development-plans/05-delivery-checklist.md`
- 测试报告：`tests/TESTING_REPORT.md`
- 运维 Runbook：`runbook.md`
