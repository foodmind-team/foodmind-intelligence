"""Scheduling module — OR-Tools CP-SAT based kitchen task scheduling.

Exports the pipeline components:
- Domain models: SchedulingProblem, ScheduleResult, etc.
- Builder: creates CP-SAT variables and constraints
- Solver: runs CP-SAT and maps statuses
- Extractor: converts solution to domain intervals
- Verifier: independent correctness check
- Orchestrator: convenience schedule() and lexicographic multi-objective

Handbook Chapter 7 reference implementation.
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
