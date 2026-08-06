"""P5-1: solve_schedule / verify_schedule 工具。"""

from __future__ import annotations

import asyncio
from typing import Any

from cooking_plan_agent.tooling.schemas import RegisteredTool


def build_solve_schedule() -> RegisteredTool:
    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        from cooking_plan_agent.domain.models import CookingTask, KitchenResourceSnapshot
        from cooking_plan_agent.scheduling.models import SchedulingProblem
        from cooking_plan_agent.scheduling.orchestrator import ScheduleOrchestrator

        problem = SchedulingProblem(
            tasks=tuple(CookingTask.model_validate(t) for t in arguments.get("tasks", ())),
            resources=tuple(KitchenResourceSnapshot.model_validate(r) for r in arguments.get("resources", ())),
            requested_time_limit_minutes=arguments.get("requested_time_limit_minutes"),
            solver_timeout_seconds=float(arguments.get("solver_timeout_seconds", 10.0)),
        )
        level = str(arguments.get("optimization_level", "full"))
        orchestrator = ScheduleOrchestrator()
        result, _ = await asyncio.to_thread(orchestrator.solve, problem, level)
        return {"schedule_result": result.model_dump(mode="json")}

    return RegisteredTool(
        name="solve_schedule",
        description="Solve a CP-SAT scheduling problem to an optimal or feasible timeline.",
        parameters={
            "type": "object",
            "properties": {
                "tasks": {"type": "array", "items": {"type": "object"}},
                "resources": {"type": "array", "items": {"type": "object"}},
                "requested_time_limit_minutes": {"type": ["integer", "null"]},
                "solver_timeout_seconds": {"type": "number"},
                "optimization_level": {"type": "string", "enum": ["full", "phase12", "makespan"]},
            },
            "required": ["tasks", "resources"],
        },
        executor=execute,
    )


def build_verify_schedule() -> RegisteredTool:
    async def execute(arguments: dict[str, Any]) -> dict[str, Any]:
        from cooking_plan_agent.scheduling.models import ScheduleResult, SchedulingProblem
        from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

        problem = SchedulingProblem.model_validate(arguments["problem"])
        result = ScheduleResult.model_validate(arguments["result"])
        report = ScheduleVerifier().verify(problem, result)
        return {"verification_report": report.model_dump(mode="json")}

    return RegisteredTool(
        name="verify_schedule",
        description="Independently verify a schedule against the original problem constraints.",
        parameters={
            "type": "object",
            "properties": {
                "problem": {"type": "object", "description": "Serialised SchedulingProblem."},
                "result": {"type": "object", "description": "Serialised ScheduleResult."},
            },
            "required": ["problem", "result"],
        },
        executor=execute,
    )
