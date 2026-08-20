# ============================================================================
# Scheduling 模块 — 基于 OR-Tools CP-SAT 的厨房任务调度
# ============================================================================

"""Scheduling module — OR-Tools CP-SAT based kitchen task scheduling.

调度模块 — 基于 OR-Tools CP-SAT 的厨房任务调度。

Exports the pipeline components:
导出流水线组件：
- Domain models: SchedulingProblem, ScheduleResult, etc.
- 领域模型：SchedulingProblem、ScheduleResult 等
- Builder: creates CP-SAT variables and constraints
- Builder（构建器）：创建 CP-SAT 变量与约束
- Solver: runs CP-SAT and maps statuses
- Solver（求解器）：运行 CP-SAT 并映射状态
- Extractor: converts solution to domain intervals
- Extractor（提取器）：将求解结果转换为领域区间
- Verifier: independent correctness check
- Verifier（校验器）：独立正确性检查
- Orchestrator: convenience schedule() and lexicographic multi-objective
- Orchestrator（编排器）：便捷的 schedule() 与字典序多目标优化
"""

from cooking_plan_agent.scheduling.builder import ModelInfo, ScheduleModelBuilder
from cooking_plan_agent.scheduling.extractor import ScheduleExtractor
from cooking_plan_agent.scheduling.models import (
    ScheduledInterval,
    ScheduleResult,
    SchedulingProblem,
    VerificationIssue,
    VerificationReport,
)
from cooking_plan_agent.scheduling.orchestrator import (
    ScheduleOrchestrator,
    schedule,
)
from cooking_plan_agent.scheduling.solver import ScheduleSolver, SolverRun
from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

__all__ = [
    "ModelInfo",
    "ScheduleExtractor",
    "ScheduleModelBuilder",
    "ScheduleOrchestrator",
    "ScheduleResult",
    "ScheduleSolver",
    "ScheduleVerifier",
    "ScheduledInterval",
    "SchedulingProblem",
    "SolverRun",
    "VerificationIssue",
    "VerificationReport",
    "schedule",
]
