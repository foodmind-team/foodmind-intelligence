# =============================================================================
# 领域枚举定义模块（domain/enums）
# -----------------------------------------------------------------------------
# 本文件集中定义烹饪计划 Agent 领域模型中的各类枚举常量，包括：
#   - PlanStatus  ：烹饪计划的声明周期状态
#   - SolverStatus：求解器结果状态（与 OR-Tools 的 CpSolverStatus 对齐）
#   - WorkMode    ：Agent 的交互模式
#   - HeatLevel   ：炉灶火力档位（用于建模烹饪任务的资源约束）
# 所有枚举均基于 StrEnum（Python 3.11+），成员值自动转换为 str，
# 便于直接与 JSON 序列化 / 反序列化、以及存储层对接。
# =============================================================================

from enum import (
    StrEnum,  # String enum base class (Python 3.11+), auto-casts member values to str
    # ↑ 字符串枚举基类（Python 3.11+），自动把成员值转换为 str
)


# Lifecycle status of a cooking plan
# 烹饪计划的声明周期状态
class PlanStatus(StrEnum):
    READY = "READY"  # Plan is generated and ready to use
    # ↑ 计划已生成，可直接使用
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"  # Plan is generated but requires user confirmation
    # ↑ 计划已生成，但需要用户确认
    INFEASIBLE = "INFEASIBLE"  # No feasible plan exists under current constraints
    # ↑ 在当前约束下不存在可行计划
    FAILED = "FAILED"  # An error occurred during scheduling; no plan produced
    # ↑ 调度过程中发生错误，未产出任何计划


# Solver result status, aligned with OR-Tools CpSolverStatus
# Important: do not collapse UNKNOWN into INFEASIBLE —
#   UNKNOWN means the solver could not determine feasibility before hitting its limit.
# 求解器结果状态，与 OR-Tools 的 CpSolverStatus 对齐
# 重要：切勿把 UNKNOWN 归并为 INFEASIBLE ——
#   UNKNOWN 表示求解器在触及限制之前未能判定可行性（而非确定不可行）。
class SolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"  # Proven optimal solution found (objective cannot be improved)
    # ↑ 已找到可证明的最优解（目标函数无法进一步优化）
    FEASIBLE = "FEASIBLE"  # Feasible solution found, but optimality not proven
    # ↑ 找到了可行解，但尚未证明其最优性
    INFEASIBLE = "INFEASIBLE"  # Problem is infeasible — no solution exists
    # ↑ 问题不可行 —— 不存在任何解
    MODEL_INVALID = "MODEL_INVALID"  # Model is malformed (contradictory constraints, bad variables, etc.)
    # ↑ 模型构建错误（约束矛盾、变量定义错误等）
    UNKNOWN = "UNKNOWN"  # Solver halted before determining feasibility (time / iteration limit)
    # ↑ 求解器在判定可行性之前即停止（因时间 / 迭代次数达到上限）


# Agent interaction mode
# Agent 的交互模式
class WorkMode(StrEnum):
    ACTIVE = "ACTIVE"  # Proactive: auto-schedule and return a plan immediately
    # ↑ 主动模式：自动调度并立即返回计划
    PASSIVE = "PASSIVE"  # Reactive: wait for user instruction, respond on demand
    # ↑ 被动模式：等待用户指令，按需响应


# Stove heat levels — used for modeling cooking-task resource constraints
# 炉灶火力档位 —— 用于建模烹饪任务的资源约束
class HeatLevel(StrEnum):
    NONE = "NONE"  # No heat required
    # ↑ 无需加热
    LOW = "LOW"  # Low heat
    # ↑ 小火
    MEDIUM = "MEDIUM"  # Medium heat
    # ↑ 中火
    HIGH = "HIGH"  # High heat
    # ↑ 大火
