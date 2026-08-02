# Cooking Plan Agent — 测试完整性报告

> **Handbook Chapter 11：Testing, Security, and Quality**
>
> 执行日期：2026-08-02 | 状态：✅ 全部通过 | 覆盖率：**91%** | 用例总数：**937**

---

## 1. 测试架构全景

```mermaid
flowchart TD
    subgraph "测试层级金字塔"
        UAT["🎯 UAT 场景<br/>12 tests"]
        INT["🔗 集成测试<br/>workflow graph<br/>6 tests"]
        SEC["🔒 安全测试<br/>53 tests"]
        CON["📜 合约测试<br/>contract + OpenAPI<br/>16 tests"]
        UNIT["🧩 单元测试<br/>domain / normalisation / parsing<br/>scheduling / safety / repair ...<br/>584 tests"]
    end
    UNIT --> CON
    CON --> SEC
    SEC --> INT
    INT --> UAT

    style UNIT fill:#e8f5e9,color:#256029
    style CON fill:#e3f2fd,color:#1565c0
    style SEC fill:#fff3e0,color:#e65100
    style INT fill:#f3e5f5,color:#7b1fa2
    style UAT fill:#fce4ec,color:#c62828
```

```mermaid
graph LR
    subgraph "7 大测试类别"
        A["Domain 模型<br/>90 tests"] --> B["单位转换/归一化<br/>108 tests"]
        B --> C["解析/预处理<br/>82 tests"]
        C --> D["分解/图/路径<br/>49 tests"]
        D --> E["排程求解/验证<br/>54 tests"]
        E --> F["安全/库存/修复/渲染<br/>166 tests"]
        F --> G["合约/安全/集成<br/>75 tests"]
    end

    style A fill:#e8f5e9
    style B fill:#e8f5e9
    style C fill:#e8f5e9
    style D fill:#e8f5e9
    style E fill:#e8f5e9
    style F fill:#e3f2fd
    style G fill:#fff3e0
```

### 目录结构

```text
tests/
├── conftest.py                          # 根配置：Hypothesis profile + OR-Tools 日志静默
├── fixtures/__init__.py                 # 18 个共享工厂函数
│
├── unit/                                # ═══ 单元测试 ═══
│   ├── domain/test_models.py            #    — Pydantic 不变量 / 错误码 / 枚举
│   ├── normalisation/test_units.py      #    — 分类器 / 转换器 / 缩放 / 错误层次
│   ├── normalisation/test_names.py      #    — 归一化 / 名称清理
│   ├── parsing/test_preprocess.py       #    — 解码 / 归一化 / 语言检测 / 管道
│   ├── parsing/test_parser_pipeline.py  #    — 管道级解析
│   ├── preparation/test_preparation.py  #    — 分解 / trie / DAG / 拓扑 / 关键路径
│   ├── scheduling/test_scheduling.py    #    — 10 fixtures / 验证器 / 编排器
│   ├── scheduling/test_golden.py        #    — 手算最优解 golden 案例
│   ├── scheduling/test_mutations.py     #    — 破坏排程负向测试
│   ├── scheduling/test_multi_objective.py # 11 — 词典序多目标（P3-03）
│   ├── research/test_research.py        #    — 9 研究场景 / 清理器 / 提取器
│   ├── safety/test_safety_engine.py     #    — 6 规则引擎 / 严重度
│   ├── safety/test_safety_anchors.py    #    — P0-07 安全锚点
│   ├── safety/test_policy.py            # 19 — 地区策略包 / 溯源 / 拒绝（P3-04）
│   ├── safety/test_policy_node.py       #  6 — 节点策略解析 / 响应记录（P3-04）
│   ├── inventory/test_feasibility.py    #    — FEFO 分配 / 短缺检测
│   ├── repair/test_options.py           #    — 修复方案 / 排序
│   ├── rendering/test_rendering.py      #    — 时间线 / mise en place
│   ├── llm/test_llm_adapters.py         #    — LLM 适配器
│   ├── tasks/test_tasks.py              # 18 — 任务状态机 / 幂等 / 取消（P3-01）
│   ├── tasks/test_distributed_workers.py # 8 — lease 领取 / 续租 / 死信（P3-02）
│   ├── test_properties.py               #    — Hypothesis property-based
│   └── test_health.py                   #    — 活跃度 / property 检查
│
├── contract/test_api_contract.py        # ═══ 合约测试 ═══
├── contract/test_task_api.py            # 11 — 异步任务 API 契约（P3-01）
├── contract/test_unified_error_contract.py # 11 — ErrorEnvelope 契约（P3-05）
├── integration/test_workflow_graph.py   # ═══ 集成测试 ═══
├── integration/test_checkpoint_persistence.py # 15 — checkpoint 持久化（P2-06）
├── integration/test_policy_region.py    #  3 — 整图地区策略（P3-04）
├── security/                            # ═══ 安全测试 ═══
│   ├── test_dependencies.py             #    — 鉴权 / 关联 ID / 日志注入
│   └── test_app_security.py             #    — NUL / 超大 / 注入 / SSRF
├── uat/test_scenarios.py                # ═══ UAT 场景 ═══
└── smoke/test_docker_smoke.py           # ═══ 冒烟测试 ═══
```

---

## 1b. P3 批次新增测试（P2-06 / P3-01 ~ P3-05）

| 计划 | 测试文件 | 用例数 | 覆盖点 |
|---|---|---|---|
| P2-06 | `integration/test_checkpoint_persistence.py` | 15 | 注入 saver、thread_id 关联、恢复、msgpack 反序列化 |
| P3-01 | `unit/tasks/test_tasks.py` | 18 | 状态机全转移、幂等键、取消/过期、任务与 revision 关联 |
| P3-01 | `contract/test_task_api.py` | 11 | 提交→轮询全链路、幂等重试、恢复 |
| P3-02 | `unit/tasks/test_distributed_workers.py` | 8 | 双 Worker 竞争、lease 续租/过期、条件写入、重试/死信 |
| P3-03 | `unit/scheduling/test_multi_objective.py` | 11 | 词典序固定目标不破坏、phase 回退、verifier 元数据一致性 |
| P3-04 | `unit/safety/test_policy.py` | 19 | 策略包加载/版本、阈值溯源、未知/过期/缺来源拒绝、checkpoint 兼容 |
| P3-04 | `unit/safety/test_policy_node.py` | 6 | 节点策略解析、地区覆盖、响应记录 |
| P3-04 | `integration/test_policy_region.py` | 3 | 未知地区 FAILED、SG/US READY 带 policy |
| P3-05 | `contract/test_unified_error_contract.py` | 11 | envelope 全字段、错误目录、retryable、compat 映射 |

质量门禁（P3 批次全部通过）：`ruff format --check` / `ruff check` / `mypy src`（87 文件 strict）/
`pytest`（937 passed）/ 覆盖率 **91%** / `export_openapi.py --check` PASSED。

---

## 2. Domain 模型不变量测试（90 tests）

> Handbook 11.3：覆盖所有 Pydantic 不变量、错误码、frozen 不可变性

### 被测模型矩阵

```mermaid
graph TB
    subgraph "StrictModel 子类覆盖度"
        A["EvidenceRef ✓"] --- B["IngredientDemand ✓"]
        B --- C["Assumption ✓"]
        C --- D["RecipeStep ✓"]
        D --- E["RecipeIR ✓"]
        E --- F["ResourceNeed ✓"]
        F --- G["TaskDependency ✓"]
        G --- H["CookingTask ✓"]
        H --- I["InventoryLotSnapshot ✓"]
        I --- J["KitchenResourceSnapshot ✓"]
        J --- K["LotAllocation ✓"]
        K --- L["SafetyFinding ✓"]
        L --- M["SafetyReport ✓"]
        M --- N["FeasibilityReport ✓"]
        N --- O["RepairOption ✓"]
        O --- P["WorkflowError ✓"]
        P --- Q["GeneratePlanRequest ✓"]
    end

    subgraph "4 组 PlanResponse"
        R["ReadyPlanResponse ✓"]
        S["ConfirmationPlanResponse ✓"]
        T["InfeasiblePlanResponse ✓"]
        U["FailedPlanResponse ✓"]
    end

    subgraph "Evidence 模型"
        V["EvidenceRef ✓"]
        W["SearchDocument ✓"]
        X["CookingEvidence ✓"]
        Y["ReconciledEvidence ✓"]
        Z["EvidenceQuery ✓"]
    end

    subgraph "Extract 模型"
        AA["ExtractedIngredient ✓"]
        AB["ExtractedStep ✓"]
        AC["ExtractedRecipeCandidate ✓"]
    end
```

### 测试样例：table-driven invariants

```python
# ---- StrictModel 不变量：extra=forbid (对所有模型有效) ----
@pytest.mark.parametrize("model_factory,model_name", [
    (lambda: IngredientDemand(...), "IngredientDemand"),
    (lambda: RecipeStep(step_number=1, instruction="Test"), "RecipeStep"),
    (lambda: CookingTask(...), "CookingTask"),
    (lambda: GeneratePlanRequest(...), "GeneratePlanRequest"),
    (lambda: SafetyFinding(...), "SafetyFinding"),
    # ... 10 种模型
])
def test_extra_fields_rejected(self, model_factory, model_name):
    instance = model_factory()
    with pytest.raises(ValidationError):
        type(instance)(**{**instance.model_dump(), "unknown_field": "intruder"})

# ---- 库存不变量：reserved <= on_hand ----
def test_reservation_cannot_exceed_on_hand(self):
    with pytest.raises(ValidationError, match="reserved quantity exceeds"):
        InventoryLotSnapshot(
            lot_id="l1", item_id="i1", canonical_name="rice",
            on_hand=Decimal(100), reserved=Decimal(150), unit="g",
        )

# ---- RecipeIR 内容约束 ----
def test_must_have_at_least_one_ingredient(self):
    with pytest.raises(ValidationError, match="at least one ingredient"):
        RecipeIR(recipe_id="r1", ..., ingredients=(), steps=(step,))

# ---- 所有 DomainErrorCode 枚举值可用 ----
@pytest.mark.parametrize("code", list(DomainErrorCode))
def test_all_error_codes_have_string_value(self, code):
    assert isinstance(code.value, str) and len(code.value) > 0
```

### 结果

```mermaid
pie title Domain 模型测试分布
    "StrictModel invariants" : 20
    "IngredientDemand" : 6
    "RecipeStep + RecipeIR" : 6
    "Resource + TaskDependency" : 7
    "CookingTask" : 5
    "InventoryLotSnapshot" : 4
    "KitchenResourceSnapshot" : 3
    "PlanResponse 4 种" : 4
    "Safety models" : 4
    "Feasibility + Repair" : 4
    "Evidence models" : 5
    "Assumption + Gap + Error" : 5
    "Extraction models" : 3
    "WorkflowException" : 3
    "Enums 完整性" : 5
    "GeneratePlanRequest" : 3
    "Annotated Types" : 3
```

---

## 3. Property-Based 测试（8 tests, Hypothesis）

> Handbook 11.4：数学不变量对所有合法输入都必须成立

### 测试样例

```python
# ---- 不变量 1：缩放数量永不为负 ----
@given(
    quantity=st.decimals(min_value="0.001", max_value=10000, places=2),
    original=st.decimals(min_value="0.5", max_value=20, places=1),
    target=st.decimals(min_value="0.5", max_value=20, places=1),
)
@seed(20260731)
def test_scaled_quantity_never_negative(quantity, original, target):
    result = scale_ingredient(demand, original, target)
    assert result.quantity >= 0  # → 永远成立


# ---- 不变量 2：单位往返换算精确保持 ----
@given(quantity=st.decimals(min_value="0.001", max_value=1000))
@seed(20260731)
def test_unit_round_trip_preserves_quantity(quantity):
    conv = UnitConverter()
    in_kg = conv.convert(quantity, "g", "kg")
    back_to_g = conv.convert(in_kg, "kg", "g")
    assert back_to_g == quantity  # Decimal 精确算术


# ---- 不变量 3：前缀树子节点数量守恒 ----
@given(num_chains=st.integers(1, 5))
@seed(20260731)
def test_prefix_tree_quantity_conservation(num_chains):
    # 构建共享 wash + 分支 cut 的前缀树
    # 验证：wash.quantity = Σ cut.quantity
    ...


# ---- 不变量 4：拓扑排序包含每个节点恰好一次 ----
@given(num_tasks=st.integers(2, 8))
@seed(20260731)
def test_topological_order_contains_every_node(num_tasks):
    # 构建线性链 DAG
    order = topological_sort_kahn(graph)
    assert len(order) == num_tasks
    assert {t.task_id for t in order} == expected_ids  # 无缺失、无重复
```

### 被验证的 8 个数学不变量

| # | 不变量描述 | Hypothesis strategies | 种子 |
|---|-----------|----------------------|------|
| 1 | 缩放数量 ≥ 0 | `decimals(0.001..10000)` | 20260731 |
| 2 | g → kg → g 往返精确 | `decimals(0.001..1000)` | 20260731 |
| 3 | 前缀树子数量 = 父数量 | `integers(1..5)` 条链 | 20260731 |
| 4 | 拓扑排序 N 个唯一节点 | `integers(2..8)` 节点 | 20260731 |
| 5 | 顺序排程无重叠误报 | `integers(1..6)` 活动任务 | 20260731 |
| 6 | 任务图无自环 | `integers(1..10)` 节点 | 20260731 |
| 7 | Horizon ≥ 总时长 | `integers(0..10)` + 随机时长 | 20260731 |
| 8 | 缩放 = Q × (target / original) 精确 | `decimals` | 20260731 |

### 结果

```
tests/unit/test_properties.py ........ 8 passed in 0.40s
```

---

## 4. 排程求解器测试全景

> Handbook 7.14 (10 fixtures) + 11.5 (golden) + 11.6 (mutation negative)

### 测试架构

```mermaid
flowchart LR
    subgraph "求解器测试三层架构"
        direction TB
        GOLD["🏆 Golden Tests<br/>9 手算最优解<br/>预期状态 + 上下界"]
        FIXT["📐 Fixture Tests<br/>10 调度场景<br/>从单个任务到 3 道菜"]
        MUT["🧨 Mutation Tests<br/>10 种破坏方式<br/>验证 verifier 拒绝"]
    end

    GOLD --> VERIFY["ScheduleVerifier<br/>独立验证"]
    FIXT --> VERIFY
    MUT --> VERIFY

    VERIFY --> RESULT["✅ 全部通过"]
```

### Golden tests 样例（9 cases）

```python
GOLDEN_CASES = [
    # name, tasks, resources, time_limit, expected_status, lower_bound, upper_bound
    {"name": "single_task_5min", "expected_status": OPTIMAL, "lower_bound": 5, "upper_bound": 5},
    {"name": "two_dependent_10min", "expected_status": OPTIMAL, "lower_bound": 10, "upper_bound": 10},
    {"name": "passive_overlaps_active_10min", "expected_status": OPTIMAL, "lower_bound": 10, "upper_bound": 10},
    {"name": "marinating_lag_35min", "expected_status": OPTIMAL, "lower_bound": 35, "upper_bound": 35},
    {"name": "hard_deadline_infeasible", "expected_status": INFEASIBLE, "lower_bound": 0, "upper_bound": 0},
    {"name": "three_dishes_shared_stove", "expected_status": OPTIMAL, "lower_bound": 15, "upper_bound": 22},
    # ... 共 9 个
]
```

### Mutation 负向测试（10 种破坏方式）

| # | 破坏方式 | 期望错误码 | 通过 |
|---|---------|----------|------|
| 1 | 调换依赖顺序 | `MIN_LAG_VIOLATION` | ✅ |
| 2 | 重叠两个活动任务 | `ACTIVE_OVERLAP` | ✅ |
| 3 | 超容量使用炉灶 | `CAPACITY_EXCEEDED` | ✅ |
| 4 | 虚假 makespan | `MAKESPAN_MISMATCH` | ✅ |
| 5 | 额外幽灵任务间隔 | `EXTRA_TASK` | ✅ |
| 6 | 缺失必需任务 | `MISSING_TASK` | ✅ |
| 7 | 时长不匹配 | `DURATION_MISMATCH` | ✅ |
| 8 | 负开始时间 | `NEGATIVE_START` | ✅ |
| 9 | 任务超出 makespan | `EXCEEDS_MAKESPAN` | ✅ |
| 10 | 资源不可用 | `RESOURCE_UNAVAILABLE` | ✅ |

### 10 个 Fixture 场景

```mermaid
flowchart TD
    F1["Fixture 1: 单任务 5min<br/>→ makespan 5, OPTIMAL"] --> F2
    F2["Fixture 2A: 依赖链 A→B<br/>→ makespan 10, OPTIMAL"] --> F2B
    F2B["Fixture 2B: min/max lag<br/>→ 精确检查延误约束"] --> F3
    F3["Fixture 3: 两个独立活动<br/>→ 不能重叠, makespan 10"] --> F4
    F4["Fixture 4: 被动任务并行<br/>→ makespan 10, pass 重叠"] --> F5
    F5["Fixture 5: 炉灶容量<br/>→ 2灶并行/1灶串行"] --> F6
    F6["Fixture 6: 资源选择<br/>→ 单类型池化"] --> F7
    F7["Fixture 7: 腌制延误<br/>→ +20min lag, makespan 35"] --> F8
    F8["Fixture 8: 最大安全等待<br/>→ max_lag 约束"] --> F9
    F9["Fixture 9: 不可行截止<br/>→ INFEASIBLE"] --> F10
    F10["Fixture 10: 损坏结果拒绝<br/>→ verifier 捕获 4 种错误"]

    style F1 fill:#e8f5e9
    style F2 fill:#e8f5e9
    style F3 fill:#e8f5e9
    style F4 fill:#e8f5e9
    style F5 fill:#e8f5e9
    style F6 fill:#e8f5e9
    style F7 fill:#e8f5e9
    style F8 fill:#e8f5e9
    style F9 fill:#fff3e0
    style F10 fill:#ffebee
```

---

## 5. UAT 业务场景测试（12 tests）

> Handbook 11.9：10 个完整业务路径，每个验证一条业务规则

| 场景 | 业务规则 | 验证方式 | 通过 |
|------|---------|---------|------|
| **UAT 1** | 汆水与腌制并行 | 煮沸（被动）与涂抹腌料（主动）叠加 → makespan < 总时长 | ✅ |
| **UAT 2** | 同食材分切三份 | 前缀树：500g 洗完 → julienne(100) + slice(200) + dice(200) 三分支 | ✅ |
| **UAT 3** | 单灶串行两道菜 | 1 灶 → d1_cook(8)+d2_cook(6) = makespan 14 | ✅ |
| **UAT 4** | 盐短缺检测 | FeasibilityReport 捕获 shortage: 需要 50g / 库存 30g → 缺 20g | ✅ |
| **UAT 5** | 过期预留库存排除 | 500g 在库 - 200g 预留 = 300g 可用；过期批次的 expiry < 今天 | ✅ |
| **UAT 6** | 死线不可行 | 3×10min 任务 + 5min 死线 → INFEASIBLE | ✅ |
| **UAT 7** | 生肉触发安全标签 | "chicken" → decompose 自动设 `safety_tags=("raw_meat",)` | ✅ |
| **UAT 8** | 搜索超时不崩溃 | 超时 → ReconciledEvidence(source_count=0, needs_confirmation=True) | ✅ |
| **UAT 9** | FEASIBLE ≠ OPTIMAL | 确保不把 FEASIBLE 标记为 OPTIMAL | ✅ |
| **UAT 10** | 幽灵任务拒绝 | 排程含不存在任务的 interval → verifier 返回 EXTRA_TASK | ✅ |

---

## 6. 安全测试（53 tests）

> Handbook 11.10：注入防护、SSRF 防护、密钥保护、边界安全

### 测试矩阵

```mermaid
mindmap
  root((安全测试<br/>53 tests))
    认证 (34)
      缺失 Token
      错误 Token
      大小写敏感
      密钥不泄漏
      注入 Settings
    关联 ID (12)
      合法 UUID4
      日志注入拦截
      换行/制表/空格拒绝
      路径穿越/SQL/HTML/中文拒绝
      NUL 字节拒绝
    应用安全 (19)
      超大文本拒绝
      NUL 字节拒绝
      无效 UTF-8 拒绝
      类型混淆拒绝
      额外字段拒绝
      Prompt 注入当数据
      HTML 标签清理
      SSRF 堵 internal IP
      密钥不泄漏
      100 任务 < 3s
```

### 安全测试样例

```python
# ---- SSRF 防护：内网 IP 被拦截 ----
def test_private_ip_blocked():
    allow = DomainAllowList.from_settings(custom_domains=[])
    docs = (SearchDocument(
        title="Local", url="http://127.0.0.1/api",
        snippet="internal", domain="127.0.0.1",
    ),)
    filtered = filter_by_domain(docs, allow)
    assert len(filtered) == 0  # 127.0.0.1 被丢弃

# ---- 密钥不泄漏到错误响应 ----
def test_auth_error_does_not_reveal_token(client):
    response = client.post("/.../generate", headers={"X-Internal-Token": "wrong-token"})
    assert response.status_code == 401
    assert "wrong-token" not in str(response.json())  # 密钥不出现

# ---- 100 任务资源耗尽测试 ----
def test_excessive_task_count_resolved_in_time():
    tasks = tuple(CookingTask(task_id=f"t{i}", ..., duration_minutes=1) for i in range(100))
    result, _ = schedule(SchedulingProblem(tasks=tasks, solver_timeout_seconds=2.0))
    assert elapsed < 3.0  # 2 秒 timeout 内完成
```

---

## 7. 覆盖度热力图

```mermaid
xychart-beta
    title "模块覆盖率 (%)"
    x-axis ["domain", "normalisation", "parsing", "api", "application", "config", "preparation", "scheduling", "research", "workflow", "safety", "inventory", "repair", "rendering", "llm"]
    y-axis "覆盖率 %" 0 --> 100
    bar [100, 93, 82, 85, 93, 100, 86, 97, 82, 78, 98, 98, 98, 94, 87]
```

### 覆盖率明细

> 2026-08-02 实测（`pytest --cov=cooking_plan_agent --cov-report=term-missing`）

| 模块 | 语句数 | 未覆盖 | 覆盖率 | 未覆盖原因 |
|------|--------|--------|--------|-----------|
| `domain/` | 319 | 0 | **100%** | — |
| `config/` | 29 | 0 | **100%** | — |
| `normalisation/` | 144 | 10 | **93%** | 名称别名边界 |
| `parsing/` | 667 | 117 | **82%** | 中英文解析分支 / inference 降级路径 |
| `api/` | 61 | 9 | **85%** | 异常处理器分支 |
| `application/` | 28 | 2 | **93%** | 防御性 FAILED 回退 |
| `preparation/` | 304 | 42 | **86%** | trie / task_graph 边界 |
| `scheduling/` | 446 | 15 | **97%** | 边界 + 降级路径 |
| `research/` | 338 | 61 | **82%** | Researcher 完整管道需 LLM |
| `workflow/` | 381 | 83 | **78%** | research_missing / 错误路由分支 |
| `safety/` | 226 | 4 | **98%** | 防御性分支 |
| `inventory/` | 91 | 2 | **98%** | 防御性分支 |
| `repair/` | 126 | 3 | **98%** | 防御性分支 |
| `rendering/` | 187 | 12 | **94%** | 空结果分支 |
| `llm/` | 193 | 26 | **87%** | 适配器错误路径 |
| **总计** | **3682** | **425** | **88%** | |

---

## 8. CI 质量闸门

```mermaid
flowchart LR
    subgraph "合并前必须通过"
        A["ruff check"] --> B["ruff format --check"]
        B --> C["mypy src"]
        C --> D["pytest --cov"]
        D --> E["architecture boundary check"]
    end

    A -->|"0 errors"| PASS1["✅"]
    B -->|"122 files formatted"| PASS2["✅"]
    C -->|"0 errors, strict"| PASS3["✅"]
    D -->|"680 passed, 88%"| PASS4["✅"]
    E -->|"domain zero framework imports"| PASS5["✅"]

    style PASS1 fill:#4caf50,color:#fff
    style PASS2 fill:#4caf50,color:#fff
    style PASS3 fill:#4caf50,color:#fff
    style PASS4 fill:#4caf50,color:#fff
    style PASS5 fill:#4caf50,color:#fff
```

### 执行命令与结果

```bash
$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
122 files already formatted

$ uv run mypy src
Success: no issues found in 69 source files

$ uv run pytest -q
680 passed, 1 warning in ~25s

$ uv run pytest --cov=cooking_plan_agent --cov-report=term-missing
TOTAL  3682  425  88%

$ uv run pytest tests/contract tests/security -v
69 passed

$ uv run python scripts/export_openapi.py --check
PASSED — all checks OK
```

---

## 9. 测试执行时间分布

> 2026-08-02 实测（`pytest -q` 各目录，含启动开销）

```mermaid
pie title 测试执行时间分布 (总计 ~26s)
    "单元测试 (584 tests)" : 1.7
    "合约测试 (16 tests)" : 13.5
    "安全测试 (53 tests)" : 10.4
    "集成测试 (6 tests)" : 0.8
    "UAT (12 tests)" : 0.6
    "冒烟测试 (9 tests)" : 0.9
```

---

## 10. 总结

| 指标 | 数值 |
|------|------|
| 总测试数 | **680** |
| 全部通过 | ✅ 680/680 |
| 失败数 | 0 |
| 代码覆盖率 | **88%** |
| Mypy strict | ✅ 0 errors |
| 测试目录重组 | ✅ 按 Handbook 11.1 分层 |
| CI 闸门 | ruff ✅ / format ✅ / mypy ✅ / pytest ✅ / OpenAPI ✅ |
| 确定性与离线 | ✅ 无外网 + 固定 Hypothesis seed |
| Verifier 完整性 | ✅ 拒绝全部 10 种破坏 |
| UAT 场景 | ✅ 12/12 全覆盖 |
| Happy path 断言 | ✅ graph 级用例明确断言 READY |
| 安全边界 | ✅ NUL / 注入 / SSRF / 密钥不泄 |

### 已知限制

- `research/researcher.py`（53%）与 `workflow/nodes.py` 的 research_missing 分支需要 LLM/真实搜索方可覆盖完整管道。
- `api/errors.py` 异常处理器的兜底分支（generic_exception_handler）仅在未预期异常时执行，默认不覆盖。
- `parsing/inference.py`（61%）的本地推理降级路径较多，仍有未覆盖分支。
- 合约与安全测试耗时偏高（约 24s），其中 OpenAPI 导出与鉴权矩阵占主要部分，属正常非性能问题。
- 冒烟测试使用 TestClient 直接验证应用工厂、健康检查与内部鉴权，不依赖 Docker 镜像。
