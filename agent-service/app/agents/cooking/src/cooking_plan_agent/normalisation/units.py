# =============================================================================
# 单位转换与份数缩放模块（normalisation/units）
# -----------------------------------------------------------------------------
# 实现手册 5.3–5.4 的“单位换算”与“份数缩放”，核心职责：
#   - UnitDimension / UnitClassifier ：把单位字符串分类到维度（质量 / 体积 / 计数）
#   - UnitConverter                  ：同维度换算（查表）+ 跨维度换算（需 ProductConversion）
#   - ProductConversion              ：跨维度桥接记录（如 1 个洋葱 ≈ 150 g）
#   - scale_ingredient               ：按份数比例线性缩放食材数量（手册 5.4）
# 设计：纯确定性函数，无 I/O，无隐式四舍五入（仅在展示边界 quantise）。
# =============================================================================

from decimal import Decimal
from enum import Enum, auto
from typing import ClassVar

from cooking_plan_agent.domain.models import (
    IngredientDemand,
    PositiveDecimal,
    StrictModel,
)
from cooking_plan_agent.normalisation.errors import (
    CrossDimensionError,
    InvalidQuantityError,
    UnitConversionError,
    UnknownUnitError,
)

# ---------------------------------------------------------------------------
# Dimension classification
# 维度分类
# ---------------------------------------------------------------------------


class UnitDimension(Enum):
    """烹饪数量换算的度量维度。

    Measurement dimensions for cooking quantity conversions.

    Members
    -------
    MASS
        Units of weight: ``mg``, ``g``, ``kg``.
        重量单位：mg / g / kg。
    VOLUME
        Units of fluid capacity: ``ml``, ``cl``, ``dl``, ``l``.
        液体容量单位：ml / cl / dl / l。
    COUNT
        Discrete item units: ``piece``, ``pc``, ``pcs``.
        离散计数单位：piece / pc / pcs。

    Conversion rules
    ----------------
    * Intra-dimension (e.g., ``g → kg``): uses fixed conversion tables
      defined at module level (``MASS_TO_GRAMS``, etc.).
    * Cross-dimension (e.g., ``piece → g``): requires a
      ``ProductConversion`` record; otherwise ``CrossDimensionError``
      is raised.

    换算规则：
    * 同维度（如 g → kg）：使用模块级固定换算表（MASS_TO_GRAMS 等）。
    * 跨维度（如 piece → g）：需要 ProductConversion 记录；否则抛 CrossDimensionError。
    """

    MASS = auto()
    VOLUME = auto()
    COUNT = auto()


# ---------------------------------------------------------------------------
# Conversion tables — canonical base unit per dimension
# 换算表 —— 每个维度的规范基准单位
# ---------------------------------------------------------------------------
# Each table maps source units to a **base-unit factor** (the quantity of
# base units per one source unit).  Intra-dimension conversion then follows:
#
#     result = quantity × factor(source) / factor(target)
#
# The base units are: gram (g), millilitre (ml), piece.
# 每张表把源单位映射到“基准单位因子”（一个源单位等于多少个基准单位）。
# 同维度换算随之遵循：result = quantity × factor(source) / factor(target)。
# 基准单位分别是：克（g）、毫升（ml）、个（piece）。

# Mass → grams conversion factors (Handbook 5.3).
# 质量 → 克 换算因子（手册 5.3）。
MASS_TO_GRAMS: dict[str, Decimal] = {
    "mg": Decimal("0.001"),
    "g": Decimal(1),
    "kg": Decimal(1000),
}

# Volume → millilitres conversion factors (Handbook 5.3).
# 体积 → 毫升 换算因子（手册 5.3）。
VOLUME_TO_ML: dict[str, Decimal] = {
    "ml": Decimal(1),
    "cl": Decimal(10),
    "dl": Decimal(100),
    "l": Decimal(1000),
}

# Count → pieces conversion factors (Handbook 5.3).
# "pc" and "pcs" are aliases that map 1:1 to "piece".
# 计数 → 个 换算因子（手册 5.3）。"pc" 与 "pcs" 是 1:1 映射到 "piece" 的别名。
COUNT_TO_PIECES: dict[str, Decimal] = {
    "piece": Decimal(1),
    "pc": Decimal(1),
    "pcs": Decimal(1),
}

# Derived index: known unit string → dimension.
# Built by merging the three conversion tables so that every recognised
# unit can be classified in O(1) time via a dict lookup.
# 派生索引：已知单位字符串 → 维度。合并三张换算表，使每个已识别单位
# 都能通过一次字典查找在 O(1) 时间内分类。
_UNIT_TO_DIMENSION: dict[str, UnitDimension] = {
    **{unit: UnitDimension.MASS for unit in MASS_TO_GRAMS},
    **{unit: UnitDimension.VOLUME for unit in VOLUME_TO_ML},
    **{unit: UnitDimension.COUNT for unit in COUNT_TO_PIECES},
}

# Derived index: dimension → conversion table.
# Allows _convert_intra to pick the right table without conditionals.
# 派生索引：维度 → 换算表。使 _convert_intra 无需条件分支即可选对表。
_DIMENSION_TO_TABLE: dict[UnitDimension, dict[str, Decimal]] = {
    UnitDimension.MASS: MASS_TO_GRAMS,
    UnitDimension.VOLUME: VOLUME_TO_ML,
    UnitDimension.COUNT: COUNT_TO_PIECES,
}

# Canonical base unit per dimension.
# Useful for display formatting and as a fallback when the output unit is
# not specified by the caller.
# 每个维度的规范基准单位。用于展示格式化，以及调用方未指定输出单位时的兜底。
_DIMENSION_BASE_UNIT: dict[UnitDimension, str] = {
    UnitDimension.MASS: "g",
    UnitDimension.VOLUME: "ml",
    UnitDimension.COUNT: "piece",
}


# ---------------------------------------------------------------------------
# ProductConversion — cross-dimension bridge
# ProductConversion —— 跨维度桥接
# ---------------------------------------------------------------------------


class ProductConversion(StrictModel):
    """产品特定的跨维度换算（不同单位维度之间）。

    A product-specific conversion between different-unit dimensions.

    Example: 1 onion ≈ 150 g.  The factor expresses the per-unit-equivalent
    in the target unit::

        quantity_in_target = quantity_in_from × conversion_factor

    例：1 个洋葱 ≈ 150 g。factor 表示每个源单位等于多少个目标单位。

    Attributes
    ----------
    canonical_name
        Normalised product identifier used as a stable key for lookup
        (e.g., ``"brown onion"``).
        用于查找稳定键的规范化产品标识（如 "brown onion"）。
    from_unit
        Source unit before conversion (e.g., ``"piece"``).
        转换前的源单位（如 "piece"）。
    to_unit
        Target unit after conversion (e.g., ``"g"``).
        转换后的目标单位（如 "g"）。
    conversion_factor
        Strictly positive ``Decimal`` expressing how many *to_units* equal
        one *from_unit*.
        严格为正的 Decimal，表示一个 from_unit 等于多少个 to_unit。
    source
        Provenance tag — ``"catalogue"`` for pre-vetted data,
        ``"user_confirmed"`` for explicit user input, ``"estimated"``
        for heuristic fallback.  Defaults to ``"catalogue"``.
        溯源标签 —— "catalogue"（预审核数据）/ "user_confirmed"（显式用户输入）
        / "estimated"（启发式兜底）。默认 "catalogue"。
    """

    canonical_name: str
    """Normalised product name (e.g. 'brown onion').

    规范化产品名（如 'brown onion'）。
    """

    from_unit: str
    """Source unit (e.g. 'piece').

    源单位（如 'piece'）。
    """

    to_unit: str
    """Target unit (e.g. 'g').

    目标单位（如 'g'）。
    """

    conversion_factor: PositiveDecimal
    """Multiplier: 1 from_unit = factor to_unit.

    乘数：1 from_unit = factor to_unit。
    """

    source: str = "catalogue"
    """Provenance: 'catalogue' | 'user_confirmed' | 'estimated'.

    溯源：'catalogue' | 'user_confirmed' | 'estimated'。
    """


# ---------------------------------------------------------------------------
# UnitClassifier — dimension lookup
# UnitClassifier —— 维度查找
# ---------------------------------------------------------------------------


class UnitClassifier:
    """无状态分类器：把单位字符串映射到度量维度。

    Stateless classifier mapping unit strings to measurement dimensions.

    All methods are ``@classmethod`` — no instantiation is needed (though
    creating an instance works for readability).

    所有方法都是 @classmethod —— 无需实例化（虽然实例化也可读）。

    Responsibilities
    ----------------
    * **classify** — resolve a raw or aliased unit string to its
      ``UnitDimension``, raising ``UnknownUnitError`` on failure.
    * **are_compatible** — shortcut to test whether two units share the
      same dimension, commonly used before bulk conversion operations.

    职责：
    * classify —— 把原始或别名单位字符串解析为其 UnitDimension，失败抛 UnknownUnitError。
    * are_compatible —— 快捷判断两个单位是否同维度，常用于批量换算前。

    Alias resolution
    ----------------
    Full-word forms (``"kilogram"``, ``"millilitre"``, ``"pieces"``, etc.)
    are normalised to their canonical abbreviation via ``_UNIT_ALIASES``
    *before* dimension lookup.  This means callers do not need to
    preprocess unit strings themselves.

    别名解析：
    全称形式（"kilogram"、"millilitre"、"pieces" 等）在维度查找之前，
    通过 _UNIT_ALIASES 规范化为规范缩写。因此调用方无需自行预处理单位字符串。

    Case sensitivity
    ----------------
    The classifier is intentionally **case-sensitive**.  For example,
    ``"KG"`` will raise ``UnknownUnitError`` because it does not match
    ``"kg"``.  Callers should lowercase inputs if case-insensitivity
    is desired.

    大小写敏感性：
    分类器刻意大小写敏感。例如 "KG" 会抛 UnknownUnitError，因为它不匹配 "kg"。
    若需大小写不敏感，调用方应先转小写。

    Usage::

        UnitClassifier.classify("kg")              # → UnitDimension.MASS
        UnitClassifier.classify("millilitre")      # → UnitDimension.VOLUME
        UnitClassifier.are_compatible("g", "kg")   # → True
        UnitClassifier.are_compatible("g", "ml")   # → False
    """

    # Full-word → canonical abbreviation mapping.
    # Applied before dimension classification so that "gram" and "g"
    # resolve identically.  If a unit is not in this map it passes
    # through unchanged to _UNIT_TO_DIMENSION.
    # 全称 → 规范缩写映射。在维度分类前应用，使 "gram" 与 "g" 解析一致。
    # 若单位不在此映射中，则原样传给 _UNIT_TO_DIMENSION。
    _UNIT_ALIASES: ClassVar[dict[str, str]] = {
        "kilogram": "kg",
        "grams": "g",
        "milligrams": "mg",
        "millilitre": "ml",
        "milliliter": "ml",
        "litre": "l",
        "liter": "l",
        "centilitre": "cl",
        "centiliter": "cl",
        "decilitre": "dl",
        "deciliter": "dl",
        "pieces": "piece",
    }

    @classmethod
    def classify(cls, unit: str) -> UnitDimension:
        """返回 *unit* 的 UnitDimension（先解析别名）。

        Return the ``UnitDimension`` for *unit*, resolving aliases first.

        Args:
            unit: A case-sensitive unit string.  Must be a non-empty
                ``str`` after stripping whitespace.
                unit：大小写敏感的单位字符串。去空白后必须是非空 str。

        Returns:
            The matching ``UnitDimension``.
            匹配的 UnitDimension。

        Raises:
            UnknownUnitError: If *unit* is ``None``, empty, whitespace-only,
                not a ``str``, or not recognised in any dimension.
            UnknownUnitError：若 unit 为 None、空、纯空白、非 str，或在任意维度未识别。
        """
        # Reject non-string or None inputs early with a clear message.
        # 及早用清晰信息拒绝非字符串或 None 输入。
        if unit is None or not isinstance(unit, str):
            raise UnknownUnitError(f"Unit must be a non-empty string, got {unit!r}")
        stripped = unit.strip()
        if not stripped:
            raise UnknownUnitError("Unit must be a non-empty string")
        # Resolve full-word aliases → canonical short form; pass-through if unknown.
        # 解析全称别名 → 规范短形式；未知则原样传递。
        normalised = cls._UNIT_ALIASES.get(stripped, stripped)
        try:
            return _UNIT_TO_DIMENSION[normalised]
        except KeyError:
            raise UnknownUnitError(f"Unknown unit {stripped!r}. Known units: {sorted(_UNIT_TO_DIMENSION)}") from None

    @classmethod
    def are_compatible(cls, unit_a: str, unit_b: str) -> bool:
        """判断两个单位是否属于同一维度。

        Check whether two units belong to the same dimension.

        Args:
            unit_a: First unit string.
                unit_a：第一个单位字符串。
            unit_b: Second unit string.
                unit_b：第二个单位字符串。

        Returns:
            ``True`` if both units share the same ``UnitDimension``,
            ``False`` otherwise.
            若两个单位共享同一 UnitDimension 则 True，否则 False。

        Raises:
            UnknownUnitError: If either unit is unrecognised.
            UnknownUnitError：若任一单位未识别。
        """
        return cls.classify(unit_a) is cls.classify(unit_b)


# ---------------------------------------------------------------------------
# UnitConverter
# 单位转换器
# ---------------------------------------------------------------------------


class UnitConverter:
    """烹饪数量的确定性单位转换器。

    Deterministic unit converter for cooking quantities.

    Intra-dimension (e.g., g → kg):
        Uses fixed conversion tables.  Rounding is *not* applied
        automatically — quantise only at the presentation boundary.

    同维度（如 g → kg）：使用固定换算表。不自动四舍五入 —— 仅在展示边界 quantise。

    Cross-dimension (e.g., piece → g):
        Requires a ``ProductConversion`` record.  Without one, raises
        ``CrossDimensionError``.

    跨维度（如 piece → g）：需要 ProductConversion 记录。没有则抛 CrossDimensionError。

    Usage::

        converter = UnitConverter()
        grams = converter.convert(Decimal("2.5"), "kg", "g")
        # → Decimal("2500")

        pc = ProductConversion(canonical_name="onion", from_unit="piece",
                               to_unit="g", conversion_factor=Decimal("150"))
        grams = converter.convert(Decimal("3"), "piece", "g", pc)
        # → Decimal("450")
    """

    def convert(
        self,
        quantity: Decimal,
        from_unit: str,
        to_unit: str,
        product_conversion: ProductConversion | None = None,
    ) -> Decimal:
        """把 *quantity* 从 *from_unit* 转换为 *to_unit*。

        Convert *quantity* from *from_unit* to *to_unit*.

        The conversion path is chosen automatically based on the dimensions
        of the two units:

        * Same dimension → intra-dimension arithmetic via
          ``_convert_intra()``.
        * Different dimensions → requires a ``ProductConversion`` record
          carrying a product-specific multiplicative factor.

        换算路径根据两个单位的维度自动选择：
        * 同维度 → 通过 _convert_intra() 做同维度算术。
        * 不同维度 → 需要携带产品特定乘数因子的 ProductConversion 记录。

        Args:
            quantity: The amount to convert.  Must be a strictly positive
                ``Decimal``.  ``float`` and ``int`` values are rejected.
                quantity：要转换的数量。必须是严格为正的 Decimal。拒绝 float 与 int。
            from_unit: Source unit string (e.g., ``"kg"``, ``"litre"``).
                from_unit：源单位字符串（如 "kg"、"litre"）。
            to_unit: Target unit string.
                to_unit：目标单位字符串。
            product_conversion: **Required** when *from_unit* and *to_unit*
                belong to different dimensions.  Must be ``None`` for
                intra-dimension conversions — supplying one there triggers
                ``UnitConversionError``.
                product_conversion：当 from_unit 与 to_unit 属于不同维度时必填。
                同维度换算必须为 None —— 提供它则触发 UnitConversionError。

        Returns:
            The converted quantity as a precise ``Decimal``.  No rounding
            is applied.
            换算后的精确 Decimal。不应用四舍五入。

        Raises:
            InvalidQuantityError: If *quantity* is not a ``Decimal``, or
                is ``<= 0``.
                InvalidQuantityError：若 quantity 不是 Decimal 或 <= 0。
            UnknownUnitError: If either *from_unit* or *to_unit* is not
                a recognised unit string.
                UnknownUnitError：若 from_unit 或 to_unit 不是已识别单位。
            CrossDimensionError: If the units belong to different dimensions
                and no *product_conversion* is provided.
                CrossDimensionError：若单位属于不同维度且未提供 product_conversion。
            UnitConversionError: If a ``ProductConversion`` is supplied for
                an intra-dimension pair, or if the record's ``from_unit`` /
                ``to_unit`` do not match the requested units.
                UnitConversionError：若为同维度对提供了 ProductConversion，或记录的
                from_unit / to_unit 与请求单位不匹配。
        """
        self._validate_quantity(quantity)

        from_dim = UnitClassifier.classify(from_unit)
        to_dim = UnitClassifier.classify(to_unit)

        # --- Intra-dimension path 同维度路径 ---
        # Both units share the same dimension; delegate to the fixed
        # conversion tables.  A ProductConversion is neither needed nor
        # permitted here — providing one is a caller error and we reject
        # it explicitly rather than silently ignoring.
        # 两个单位同维度；委托给固定换算表。此处既不需要也不允许 ProductConversion ——
        # 提供它是调用方错误，我们显式拒绝而非静默忽略。
        if from_dim is to_dim:
            if product_conversion is not None:
                raise UnitConversionError(
                    f"ProductConversion must not be supplied for intra-dimension conversion ({from_unit} → {to_unit})"
                )
            return self._convert_intra(quantity, from_unit, to_unit, from_dim)

        # --- Cross-dimension path 跨维度路径 ---
        # Units differ in dimension.  A ProductConversion record is
        # mandatory to bridge the gap (Handbook 5.3).
        # 单位维度不同。必须用 ProductConversion 记录来桥接差距（手册 5.3）。
        if product_conversion is None:
            raise CrossDimensionError(
                f"Cannot convert {from_unit} ({from_dim.name}) → "
                f"{to_unit} ({to_dim.name}) without a ProductConversion. "
                "See handbook 5.3."
            )

        # Guard: the supplied ProductConversion must match exactly the
        # from_unit / to_unit requested by the caller.  A mismatch would
        # silently produce wrong results, so we fail loudly.
        # 守卫：提供的 ProductConversion 必须精确匹配调用方请求的 from_unit / to_unit。
        # 不匹配会静默产生错误结果，因此我们大声失败。
        if product_conversion.from_unit != from_unit:
            raise UnitConversionError(
                f"ProductConversion from_unit {product_conversion.from_unit!r} "
                f"does not match requested from_unit {from_unit!r}"
            )
        if product_conversion.to_unit != to_unit:
            raise UnitConversionError(
                f"ProductConversion to_unit {product_conversion.to_unit!r} does not match requested to_unit {to_unit!r}"
            )

        # Cross-dimension conversion is a simple linear multiplication:
        #   quantity_in_target = quantity_in_from × conversion_factor
        # 跨维度换算是简单线性乘法：quantity_in_target = quantity_in_from × conversion_factor
        return quantity * product_conversion.conversion_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_quantity(quantity: Decimal) -> None:
        """及早拒绝非正或非 Decimal 的数量。

        Reject non-positive or non-``Decimal`` quantities early.

        This guard runs before any dimension classification or conversion
        logic so that callers get a consistent ``InvalidQuantityError``
        regardless of which conversion path would have been taken.

        该守卫在任何维度分类或换算逻辑之前运行，使调用方无论走哪条换算路径，
        都能得到一致的 InvalidQuantityError。

        Args:
            quantity: The value to validate.
                quantity：要校验的值。

        Raises:
            InvalidQuantityError: If *quantity* is not an instance of
                ``Decimal``, or if it is ``<= 0``.
                InvalidQuantityError：若 quantity 不是 Decimal 实例或 <= 0。
        """
        if not isinstance(quantity, Decimal):
            raise InvalidQuantityError(f"Quantity must be a Decimal, got {type(quantity).__name__}")
        if quantity <= 0:
            raise InvalidQuantityError(f"Quantity must be > 0, got {quantity}")

    @staticmethod
    def _convert_intra(
        quantity: Decimal,
        from_unit: str,
        to_unit: str,
        dimension: UnitDimension,
    ) -> Decimal:
        """在同 *dimension* 的两个单位之间转换 *quantity*。

        Convert *quantity* between two units of the same *dimension*.

        Strategy
        --------
        ``from_unit → base_unit → to_unit``, using the dimension's
        conversion table so that **every** pair of units within a dimension
        is handled by a single lookup table.

        策略：
        from_unit → base_unit → to_unit，使用该维度的换算表，使维度内每一对单位
        都由同一张查找表处理。

        Formula::

            result = quantity × factor(from_unit) / factor(to_unit)

        Alias resolution (e.g., ``"litre" → "l"``) is applied before
        table lookup so that callers do not need to pre-normalise units.

        别名解析（如 "litre" → "l"）在表查找前应用，因此调用方无需预规范化单位。

        Args:
            quantity: The amount to convert (already validated ``> 0``).
                quantity：要转换的数量（已校验 > 0）。
            from_unit: Source unit string (may include aliases).
                from_unit：源单位字符串（可含别名）。
            to_unit: Target unit string (may include aliases).
                to_unit：目标单位字符串（可含别名）。
            dimension: The shared ``UnitDimension`` of both units.
                dimension：两个单位共享的 UnitDimension。

        Returns:
            The converted quantity as a ``Decimal``.
            换算后的 Decimal。
        """
        table = _DIMENSION_TO_TABLE[dimension]
        # Resolve full-word aliases to canonical short forms before
        # performing the table lookup.
        # 在执行表查找前，把全称别名解析为规范短形式。
        normalised_from = UnitClassifier._UNIT_ALIASES.get(from_unit, from_unit)
        normalised_to = UnitClassifier._UNIT_ALIASES.get(to_unit, to_unit)
        # Look up conversion factors relative to the dimension's base unit.
        # 查找相对于该维度基准单位的换算因子。
        to_base = table[normalised_from]  # factor: from_unit → base
        from_base = table[normalised_to]  # factor: to_unit → base
        # quantity_in_base = quantity × to_base
        # result = quantity_in_base / from_base
        return quantity * to_base / from_base


# ---------------------------------------------------------------------------
# 5.4 Serving scaling
# 5.4 份数缩放
# ---------------------------------------------------------------------------


def scale_ingredient(
    demand: IngredientDemand,
    original_servings: Decimal,
    target_servings: Decimal,
) -> IngredientDemand:
    """把食材需求按原始份数到目标份数做线性缩放。

    Linearly scale an ingredient demand from original to target servings.

    Implements Handbook 5.4 serving-scaling formula::

        λ = target_servings / original_servings
        Q_new = Q_original × λ

    实现手册 5.4 份数缩放公式：λ = target / original，Q_new = Q_original × λ。

    Where *λ* (lambda) is the uniform scaling factor derived from the
    servings ratio, and *Q_new* is the proportionally adjusted quantity.

    其中 λ（lambda）是由份数比导出的统一缩放因子，Q_new 是按比例调整后的数量。

    Design
    ------
    * **Immutable**: ``IngredientDemand`` is a frozen Pydantic model.
      This function uses ``model_copy(update={...})`` to produce a new
      instance, leaving the input unchanged — safe for concurrent reads
      from multiple recipe scaling passes.
    * **Linear only**: scaling is purely proportional.  Non-linear effects
      (e.g., seasoning that does not double with servings) are not modelled
      here and must be handled upstream by the recipe parser.

    设计：
    * 不可变：IngredientDemand 是 frozen Pydantic 模型。本函数用 model_copy(update={...})
      生成新实例，输入保持不变 —— 对多次菜谱缩放的并发读取安全。
    * 仅线性：缩放纯比例。非线性效应（如调味料不随份数翻倍）此处不建模，须由上游菜谱解析器处理。

    Args:
        demand: A validated ``IngredientDemand`` carrying ``quantity > 0``.
            demand：已校验、quantity > 0 的 IngredientDemand。
        original_servings: The number of servings the recipe was written for.
            Must be ``> 0``.
            original_servings：菜谱原本的份数。必须 > 0。
        target_servings: The desired number of servings.  Must be ``> 0``.
            target_servings：期望的份数。必须 > 0。

    Returns:
        A new ``IngredientDemand`` identical to *demand* except for
        ``quantity``, which is scaled by ``target / original``.
        与 demand 相同、但 quantity 按 target / original 缩放的新 IngredientDemand。

    Raises:
        InvalidQuantityError: If *original_servings* or *target_servings*
            is ``<= 0``.
            InvalidQuantityError：若 original_servings 或 target_servings <= 0。
    """
    if original_servings <= 0:
        raise InvalidQuantityError(f"original_servings must be > 0, got {original_servings}")
    if target_servings <= 0:
        raise InvalidQuantityError(f"target_servings must be > 0, got {target_servings}")

    lam = target_servings / original_servings
    new_quantity = demand.quantity * lam

    # IngredientDemand is frozen (immutable).  Use model_copy to produce
    # a new instance with only the quantity field updated; all other fields
    # (canonical_name, unit, confidence, evidence, etc.) carry through unchanged.
    # IngredientDemand 是 frozen（不可变）。用 model_copy 生成新实例，只更新 quantity 字段；
    # 其他字段（canonical_name、unit、confidence、evidence 等）原样保留。
    return demand.model_copy(update={"quantity": new_quantity})
