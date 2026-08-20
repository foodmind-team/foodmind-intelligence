# =============================================================================
# 单位转换异常模块（normalisation/errors）
# -----------------------------------------------------------------------------
# 定义单位转换相关的异常层级，用于在“规范化 / 单位换算”阶段把失败原因
# 结构化地区分开，供上层按异常类型做精确处理。
#   - UnitConversionError   ：单位转换的基类异常
#   - UnknownUnitError      ：单位字符串在任意维度中都未被识别
#   - CrossDimensionError   ：跨维度转换但缺少 ProductConversion 记录
#   - InvalidQuantityError  ：待转换的数量为零或负数
# =============================================================================

class UnitConversionError(ValueError):
    """单位转换无法执行时抛出。

    Raised when a unit conversion cannot be performed.

    Covers invalid units, cross-dimension without ProductConversion,
    and zero/negative quantities.

    涵盖：非法单位、跨维度但无 ProductConversion、以及零 / 负数量。
    """


class UnknownUnitError(UnitConversionError):
    """单位字符串在任意维度中都未被识别时抛出。

    Raised when a unit string is not recognised in any dimension.
    """


class CrossDimensionError(UnitConversionError):
    """在没有 ProductConversion 记录的情况下尝试跨维度转换时抛出。

    Raised when attempting cross-dimension conversion without a
    ProductConversion record.
    """


class InvalidQuantityError(UnitConversionError):
    """待转换的数量为零或负数时抛出。

    Raised when the quantity to convert is zero or negative.
    """
