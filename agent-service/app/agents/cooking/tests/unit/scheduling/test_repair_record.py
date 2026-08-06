"""P5-3: RepairAttemptRecord 模型与 PlanState 字段。"""
from cooking_plan_agent.scheduling.models import RepairAttemptRecord


def test_repair_record_default_detail():
    record = RepairAttemptRecord(
        attempt=1,
        issues=("CAPACITY_EXCEEDED",),
        action="lower_optimization_level",
        outcome="retrying",
    )
    assert record.attempt == 1
    assert record.outcome == "retrying"


def test_repair_record_is_serialisable():
    record = RepairAttemptRecord(
        attempt=2, issues=("MISSING_TASK",), action="give_up", outcome="gave_up"
    )
    dumped = record.model_dump()
    assert dumped["issues"] == ("MISSING_TASK",)
