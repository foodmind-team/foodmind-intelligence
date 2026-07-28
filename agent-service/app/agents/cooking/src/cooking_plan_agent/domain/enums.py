from enum import StrEnum  # String enum base class (Python 3.11+), auto-casts member values to str

# Lifecycle status of a cooking plan
class PlanStatus(StrEnum):
    READY = "READY"                                # Plan is generated and ready to use
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"      # Plan is generated but requires user confirmation
    INFEASIBLE = "INFEASIBLE"                      # No feasible plan exists under current constraints
    FAILED = "FAILED"                              # An error occurred during scheduling; no plan produced

# Solver result status, aligned with OR-Tools CpSolverStatus
# Important: do not collapse UNKNOWN into INFEASIBLE —
#   UNKNOWN means the solver could not determine feasibility before hitting its limit.
class SolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"              # Proven optimal solution found (objective cannot be improved)
    FEASIBLE = "FEASIBLE"            # Feasible solution found, but optimality not proven
    INFEASIBLE = "INFEASIBLE"        # Problem is infeasible — no solution exists
    MODEL_INVALID = "MODEL_INVALID"  # Model is malformed (contradictory constraints, bad variables, etc.)
    UNKNOWN = "UNKNOWN"              # Solver halted before determining feasibility (time / iteration limit)

# Agent interaction mode
class WorkMode(StrEnum):
    ACTIVE = "ACTIVE"    # Proactive: auto-schedule and return a plan immediately
    PASSIVE = "PASSIVE"  # Reactive: wait for user instruction, respond on demand

# Stove heat levels — used for modeling cooking-task resource constraints
class HeatLevel(StrEnum):
    NONE = "NONE"        # No heat required
    LOW = "LOW"          # Low heat
    MEDIUM = "MEDIUM"    # Medium heat
    HIGH = "HIGH"        # High heat
