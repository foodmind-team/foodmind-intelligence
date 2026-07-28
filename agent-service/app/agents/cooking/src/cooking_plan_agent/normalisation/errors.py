class UnitConversionError(ValueError):
    """Raised when a unit conversion cannot be performed.

    Covers invalid units, cross-dimension without ProductConversion,
    and zero/negative quantities.
    """


class UnknownUnitError(UnitConversionError):
    """Raised when a unit string is not recognised in any dimension."""


class CrossDimensionError(UnitConversionError):
    """Raised when attempting cross-dimension conversion without a
    ProductConversion record."""


class InvalidQuantityError(UnitConversionError):
    """Raised when the quantity to convert is zero or negative."""
