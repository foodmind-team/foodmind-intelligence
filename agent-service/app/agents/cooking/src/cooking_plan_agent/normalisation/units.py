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
# ---------------------------------------------------------------------------


class UnitDimension(Enum):
    """Measurement dimensions for cooking quantity conversions.

    Members
    -------
    MASS
        Units of weight: ``mg``, ``g``, ``kg``.
    VOLUME
        Units of fluid capacity: ``ml``, ``cl``, ``dl``, ``l``.
    COUNT
        Discrete item units: ``piece``, ``pc``, ``pcs``.

    Conversion rules
    ----------------
    * Intra-dimension (e.g., ``g → kg``): uses fixed conversion tables
      defined at module level (``MASS_TO_GRAMS``, etc.).
    * Cross-dimension (e.g., ``piece → g``): requires a
      ``ProductConversion`` record; otherwise ``CrossDimensionError``
      is raised.
    """

    MASS = auto()
    VOLUME = auto()
    COUNT = auto()


# ---------------------------------------------------------------------------
# Conversion tables — canonical base unit per dimension
# ---------------------------------------------------------------------------
# Each table maps source units to a **base-unit factor** (the quantity of
# base units per one source unit).  Intra-dimension conversion then follows:
#
#     result = quantity × factor(source) / factor(target)
#
# The base units are: gram (g), millilitre (ml), piece.

# Mass → grams conversion factors (Handbook 5.3).
MASS_TO_GRAMS: dict[str, Decimal] = {
    "mg": Decimal("0.001"),
    "g": Decimal(1),
    "kg": Decimal(1000),
}

# Volume → millilitres conversion factors (Handbook 5.3).
VOLUME_TO_ML: dict[str, Decimal] = {
    "ml": Decimal(1),
    "cl": Decimal(10),
    "dl": Decimal(100),
    "l": Decimal(1000),
}

# Count → pieces conversion factors (Handbook 5.3).
# "pc" and "pcs" are aliases that map 1:1 to "piece".
COUNT_TO_PIECES: dict[str, Decimal] = {
    "piece": Decimal(1),
    "pc": Decimal(1),
    "pcs": Decimal(1),
}

# Derived index: known unit string → dimension.
# Built by merging the three conversion tables so that every recognised
# unit can be classified in O(1) time via a dict lookup.
_UNIT_TO_DIMENSION: dict[str, UnitDimension] = {
    **{unit: UnitDimension.MASS for unit in MASS_TO_GRAMS},
    **{unit: UnitDimension.VOLUME for unit in VOLUME_TO_ML},
    **{unit: UnitDimension.COUNT for unit in COUNT_TO_PIECES},
}

# Derived index: dimension → conversion table.
# Allows _convert_intra to pick the right table without conditionals.
_DIMENSION_TO_TABLE: dict[UnitDimension, dict[str, Decimal]] = {
    UnitDimension.MASS: MASS_TO_GRAMS,
    UnitDimension.VOLUME: VOLUME_TO_ML,
    UnitDimension.COUNT: COUNT_TO_PIECES,
}

# Canonical base unit per dimension.
# Useful for display formatting and as a fallback when the output unit is
# not specified by the caller.
_DIMENSION_BASE_UNIT: dict[UnitDimension, str] = {
    UnitDimension.MASS: "g",
    UnitDimension.VOLUME: "ml",
    UnitDimension.COUNT: "piece",
}


# ---------------------------------------------------------------------------
# ProductConversion — cross-dimension bridge
# ---------------------------------------------------------------------------


class ProductConversion(StrictModel):
    """A product-specific conversion between different-unit dimensions.

    Example: 1 onion ≈ 150 g.  The factor expresses the per-unit-equivalent
    in the target unit::

        quantity_in_target = quantity_in_from × conversion_factor

    Attributes
    ----------
    canonical_name
        Normalised product identifier used as a stable key for lookup
        (e.g., ``"brown onion"``).
    from_unit
        Source unit before conversion (e.g., ``"piece"``).
    to_unit
        Target unit after conversion (e.g., ``"g"``).
    conversion_factor
        Strictly positive ``Decimal`` expressing how many *to_units* equal
        one *from_unit*.
    source
        Provenance tag — ``"catalogue"`` for pre-vetted data,
        ``"user_confirmed"`` for explicit user input, ``"estimated"``
        for heuristic fallback.  Defaults to ``"catalogue"``.

    """

    canonical_name: str
    """Normalised product name (e.g. 'brown onion')."""

    from_unit: str
    """Source unit (e.g. 'piece')."""

    to_unit: str
    """Target unit (e.g. 'g')."""

    conversion_factor: PositiveDecimal
    """Multiplier: 1 from_unit = factor to_unit."""

    source: str = "catalogue"
    """Provenance: 'catalogue' | 'user_confirmed' | 'estimated'."""


# ---------------------------------------------------------------------------
# UnitClassifier — dimension lookup
# ---------------------------------------------------------------------------


class UnitClassifier:
    """Stateless classifier mapping unit strings to measurement dimensions.

    All methods are ``@classmethod`` — no instantiation is needed (though
    creating an instance works for readability).

    Responsibilities
    ----------------
    * **classify** — resolve a raw or aliased unit string to its
      ``UnitDimension``, raising ``UnknownUnitError`` on failure.
    * **are_compatible** — shortcut to test whether two units share the
      same dimension, commonly used before bulk conversion operations.

    Alias resolution
    ----------------
    Full-word forms (``"kilogram"``, ``"millilitre"``, ``"pieces"``, etc.)
    are normalised to their canonical abbreviation via ``_UNIT_ALIASES``
    *before* dimension lookup.  This means callers do not need to
    preprocess unit strings themselves.

    Case sensitivity
    ----------------
    The classifier is intentionally **case-sensitive**.  For example,
    ``"KG"`` will raise ``UnknownUnitError`` because it does not match
    ``"kg"``.  Callers should lowercase inputs if case-insensitivity
    is desired.

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
        """Return the ``UnitDimension`` for *unit*, resolving aliases first.

        Args:
            unit: A case-sensitive unit string.  Must be a non-empty
                ``str`` after stripping whitespace.

        Returns:
            The matching ``UnitDimension``.

        Raises:
            UnknownUnitError: If *unit* is ``None``, empty, whitespace-only,
                not a ``str``, or not recognised in any dimension.
        """
        # Reject non-string or None inputs early with a clear message.
        if unit is None or not isinstance(unit, str):
            raise UnknownUnitError(f"Unit must be a non-empty string, got {unit!r}")
        stripped = unit.strip()
        if not stripped:
            raise UnknownUnitError("Unit must be a non-empty string")
        # Resolve full-word aliases → canonical short form; pass-through if unknown.
        normalised = cls._UNIT_ALIASES.get(stripped, stripped)
        try:
            return _UNIT_TO_DIMENSION[normalised]
        except KeyError:
            raise UnknownUnitError(
                f"Unknown unit {stripped!r}. Known units: "
                f"{sorted(_UNIT_TO_DIMENSION)}"
            ) from None

    @classmethod
    def are_compatible(cls, unit_a: str, unit_b: str) -> bool:
        """Check whether two units belong to the same dimension.

        Args:
            unit_a: First unit string.
            unit_b: Second unit string.

        Returns:
            ``True`` if both units share the same ``UnitDimension``,
            ``False`` otherwise.

        Raises:
            UnknownUnitError: If either unit is unrecognised.
        """
        return cls.classify(unit_a) is cls.classify(unit_b)


# ---------------------------------------------------------------------------
# UnitConverter
# ---------------------------------------------------------------------------


class UnitConverter:
    """Deterministic unit converter for cooking quantities.

    Intra-dimension (e.g., g → kg):
        Uses fixed conversion tables.  Rounding is *not* applied
        automatically — quantise only at the presentation boundary.

    Cross-dimension (e.g., piece → g):
        Requires a ``ProductConversion`` record.  Without one, raises
        ``CrossDimensionError``.

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
        """Convert *quantity* from *from_unit* to *to_unit*.

        The conversion path is chosen automatically based on the dimensions
        of the two units:

        * Same dimension → intra-dimension arithmetic via
          ``_convert_intra()``.
        * Different dimensions → requires a ``ProductConversion`` record
          carrying a product-specific multiplicative factor.

        Args:
            quantity: The amount to convert.  Must be a strictly positive
                ``Decimal``.  ``float`` and ``int`` values are rejected.
            from_unit: Source unit string (e.g., ``"kg"``, ``"litre"``).
            to_unit: Target unit string.
            product_conversion: **Required** when *from_unit* and *to_unit*
                belong to different dimensions.  Must be ``None`` for
                intra-dimension conversions — supplying one there triggers
                ``UnitConversionError``.

        Returns:
            The converted quantity as a precise ``Decimal``.  No rounding
            is applied.

        Raises:
            InvalidQuantityError: If *quantity* is not a ``Decimal``, or
                is ``<= 0``.
            UnknownUnitError: If either *from_unit* or *to_unit* is not
                a recognised unit string.
            CrossDimensionError: If the units belong to different dimensions
                and no *product_conversion* is provided.
            UnitConversionError: If a ``ProductConversion`` is supplied for
                an intra-dimension pair, or if the record's ``from_unit`` /
                ``to_unit`` do not match the requested units.
        """
        self._validate_quantity(quantity)

        from_dim = UnitClassifier.classify(from_unit)
        to_dim = UnitClassifier.classify(to_unit)

        # --- Intra-dimension path ---
        # Both units share the same dimension; delegate to the fixed
        # conversion tables.  A ProductConversion is neither needed nor
        # permitted here — providing one is a caller error and we reject
        # it explicitly rather than silently ignoring.
        if from_dim is to_dim:
            if product_conversion is not None:
                raise UnitConversionError(
                    "ProductConversion must not be supplied for "
                    f"intra-dimension conversion ({from_unit} → {to_unit})"
                )
            return self._convert_intra(quantity, from_unit, to_unit, from_dim)

        # --- Cross-dimension path ---
        # Units differ in dimension.  A ProductConversion record is
        # mandatory to bridge the gap (Handbook 5.3).
        if product_conversion is None:
            raise CrossDimensionError(
                f"Cannot convert {from_unit} ({from_dim.name}) → "
                f"{to_unit} ({to_dim.name}) without a ProductConversion. "
                "See handbook 5.3."
            )

        # Guard: the supplied ProductConversion must match exactly the
        # from_unit / to_unit requested by the caller.  A mismatch would
        # silently produce wrong results, so we fail loudly.
        if product_conversion.from_unit != from_unit:
            raise UnitConversionError(
                f"ProductConversion from_unit {product_conversion.from_unit!r} "
                f"does not match requested from_unit {from_unit!r}"
            )
        if product_conversion.to_unit != to_unit:
            raise UnitConversionError(
                f"ProductConversion to_unit {product_conversion.to_unit!r} "
                f"does not match requested to_unit {to_unit!r}"
            )

        # Cross-dimension conversion is a simple linear multiplication:
        #   quantity_in_target = quantity_in_from × conversion_factor
        return quantity * product_conversion.conversion_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_quantity(quantity: Decimal) -> None:
        """Reject non-positive or non-``Decimal`` quantities early.

        This guard runs before any dimension classification or conversion
        logic so that callers get a consistent ``InvalidQuantityError``
        regardless of which conversion path would have been taken.

        Args:
            quantity: The value to validate.

        Raises:
            InvalidQuantityError: If *quantity* is not an instance of
                ``Decimal``, or if it is ``<= 0``.
        """
        if not isinstance(quantity, Decimal):
            raise InvalidQuantityError(
                f"Quantity must be a Decimal, got {type(quantity).__name__}"
            )
        if quantity <= 0:
            raise InvalidQuantityError(
                f"Quantity must be > 0, got {quantity}"
            )

    @staticmethod
    def _convert_intra(
        quantity: Decimal,
        from_unit: str,
        to_unit: str,
        dimension: UnitDimension,
    ) -> Decimal:
        """Convert *quantity* between two units of the same *dimension*.

        Strategy
        --------
        ``from_unit → base_unit → to_unit``, using the dimension's
        conversion table so that **every** pair of units within a dimension
        is handled by a single lookup table.

        Formula::

            result = quantity × factor(from_unit) / factor(to_unit)

        Alias resolution (e.g., ``"litre" → "l"``) is applied before
        table lookup so that callers do not need to pre-normalise units.

        Args:
            quantity: The amount to convert (already validated ``> 0``).
            from_unit: Source unit string (may include aliases).
            to_unit: Target unit string (may include aliases).
            dimension: The shared ``UnitDimension`` of both units.

        Returns:
            The converted quantity as a ``Decimal``.
        """
        table = _DIMENSION_TO_TABLE[dimension]
        # Resolve full-word aliases to canonical short forms before
        # performing the table lookup.
        normalised_from = UnitClassifier._UNIT_ALIASES.get(from_unit, from_unit)
        normalised_to = UnitClassifier._UNIT_ALIASES.get(to_unit, to_unit)
        # Look up conversion factors relative to the dimension's base unit.
        to_base = table[normalised_from]      # factor: from_unit → base
        from_base = table[normalised_to]      # factor: to_unit → base
        # quantity_in_base = quantity × to_base
        # result = quantity_in_base / from_base
        return quantity * to_base / from_base


# ---------------------------------------------------------------------------
# 5.4 Serving scaling
# ---------------------------------------------------------------------------


def scale_ingredient(
    demand: IngredientDemand,
    original_servings: Decimal,
    target_servings: Decimal,
) -> IngredientDemand:
    """Linearly scale an ingredient demand from original to target servings.

    Implements Handbook 5.4 serving-scaling formula::

        λ = target_servings / original_servings
        Q_new = Q_original × λ

    Where *λ* (lambda) is the uniform scaling factor derived from the
    servings ratio, and *Q_new* is the proportionally adjusted quantity.

    Design
    ------
    * **Immutable**: ``IngredientDemand`` is a frozen Pydantic model.
      This function uses ``model_copy(update={...})`` to produce a new
      instance, leaving the input unchanged — safe for concurrent reads
      from multiple recipe scaling passes.
    * **Linear only**: scaling is purely proportional.  Non-linear effects
      (e.g., seasoning that does not double with servings) are not modelled
      here and must be handled upstream by the recipe parser.

    Args:
        demand: A validated ``IngredientDemand`` carrying ``quantity > 0``.
        original_servings: The number of servings the recipe was written for.
            Must be ``> 0``.
        target_servings: The desired number of servings.  Must be ``> 0``.

    Returns:
        A new ``IngredientDemand`` identical to *demand* except for
        ``quantity``, which is scaled by ``target / original``.

    Raises:
        InvalidQuantityError: If *original_servings* or *target_servings*
            is ``<= 0``.
    """
    if original_servings <= 0:
        raise InvalidQuantityError(
            f"original_servings must be > 0, got {original_servings}"
        )
    if target_servings <= 0:
        raise InvalidQuantityError(
            f"target_servings must be > 0, got {target_servings}"
        )

    lam = target_servings / original_servings
    new_quantity = demand.quantity * lam

    # IngredientDemand is frozen (immutable).  Use model_copy to produce
    # a new instance with only the quantity field updated; all other fields
    # (canonical_name, unit, confidence, evidence, etc.) carry through unchanged.
    return demand.model_copy(update={"quantity": new_quantity})
