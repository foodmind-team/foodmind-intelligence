# =============================================================================
# 修复服务的小型值对象模块（repair/models）
# -----------------------------------------------------------------------------
# 定义修复服务使用的小型值对象：
#   - Shortage         ：单个资源或食材短缺（NamedTuple）
#   - RepairValidation ：单个 RepairOption 的校验结果（StrictModel）
# =============================================================================

"""Small value objects used by repair services.

修复服务使用的小型值对象。
"""

from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.models import StrictModel


class Shortage(NamedTuple):
    """A single resource or ingredient shortage.

    单个资源或食材短缺。
    """

    item: str
    """Ingredient name or resource type.

    食材名称或资源类型。
    """
    required: Decimal
    """Amount needed.

    所需数量。
    """
    available: Decimal
    """Amount available.

    可用数量。
    """
    unit: str
    """Unit of measure.

    计量单位。
    """


class RepairValidation(StrictModel):
    """Result of validating a single RepairOption.

    校验单个 RepairOption 的结果。
    """

    is_valid: bool
    """Whether the option is internally consistent.

    该选项是否内部一致。
    """
    issues: tuple[str, ...] = ()
    """Validation issues, if any.

    校验问题（如有）。
    """


# =============================================================================
# 5.17  Calculate exact shortages
#      计算精确短缺量
# =============================================================================
