from decimal import Decimal

import pytest

from cooking_plan_agent.domain.models import (
    IngredientDemand,
)
from cooking_plan_agent.normalisation.errors import (
    CrossDimensionError,
    InvalidQuantityError,
    UnitConversionError,
    UnknownUnitError,
)
from cooking_plan_agent.normalisation.units import (
    COUNT_TO_PIECES,
    MASS_TO_GRAMS,
    VOLUME_TO_ML,
    ProductConversion,
    UnitClassifier,
    UnitConverter,
    UnitDimension,
    scale_ingredient,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def converter() -> UnitConverter:
    """A fresh UnitConverter instance for each test."""
    return UnitConverter()


@pytest.fixture
def onion_conversion() -> ProductConversion:
    """A valid product conversion: 1 onion ≈ 150 g."""
    return ProductConversion(
        canonical_name="brown onion",
        from_unit="piece",
        to_unit="g",
        conversion_factor=Decimal(150),
        source="catalogue",
    )


@pytest.fixture
def sample_demand() -> IngredientDemand:
    """A minimal valid IngredientDemand for scale_ingredient tests."""
    return IngredientDemand(
        canonical_name="chicken breast",
        raw_name="chicken breast",
        quantity=Decimal(500),
        unit="g",
        confidence=Decimal("0.95"),
    )


# ======================================================================
# UnitDimension
# ======================================================================


class TestUnitDimension:
    """UnitDimension enum members and uniqueness."""

    def test_distinct_members(self) -> None:
        # All three dimensions must be different enum values.
        assert UnitDimension.MASS != UnitDimension.VOLUME
        assert UnitDimension.VOLUME != UnitDimension.COUNT
        assert UnitDimension.MASS != UnitDimension.COUNT

    def test_length(self) -> None:
        # Only three cooking dimensions are defined.
        assert len(UnitDimension) == 3


# ======================================================================
# Conversion tables — structural checks
# ======================================================================


class TestConversionTables:
    """Verify the conversion tables are well-formed."""

    def test_mass_table_keys(self) -> None:
        assert "mg" in MASS_TO_GRAMS
        assert "g" in MASS_TO_GRAMS
        assert "kg" in MASS_TO_GRAMS

    def test_volume_table_keys(self) -> None:
        assert "ml" in VOLUME_TO_ML
        assert "cl" in VOLUME_TO_ML
        assert "dl" in VOLUME_TO_ML
        assert "l" in VOLUME_TO_ML

    def test_count_table_keys(self) -> None:
        assert "piece" in COUNT_TO_PIECES
        assert "pc" in COUNT_TO_PIECES
        assert "pcs" in COUNT_TO_PIECES

    def test_all_factors_are_positive_decimals(self) -> None:
        for table in (MASS_TO_GRAMS, VOLUME_TO_ML, COUNT_TO_PIECES):
            for unit, factor in table.items():
                assert isinstance(factor, Decimal), f"{unit} factor not Decimal"
                assert factor > 0, f"{unit} factor {factor} <= 0"


# ======================================================================
# UnitClassifier.classify
# ======================================================================


class TestClassifyHappy:
    """Happy-path classify: all known units map to correct dimensions."""

    @pytest.mark.parametrize(
        "unit,expected",
        [
            ("mg", UnitDimension.MASS),
            ("g", UnitDimension.MASS),
            ("kg", UnitDimension.MASS),
            ("ml", UnitDimension.VOLUME),
            ("cl", UnitDimension.VOLUME),
            ("dl", UnitDimension.VOLUME),
            ("l", UnitDimension.VOLUME),
            ("piece", UnitDimension.COUNT),
            ("pc", UnitDimension.COUNT),
            ("pcs", UnitDimension.COUNT),
        ],
    )
    def test_known_unit(self, unit: str, expected: UnitDimension) -> None:
        assert UnitClassifier.classify(unit) is expected

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("kilogram", UnitDimension.MASS),
            ("grams", UnitDimension.MASS),
            ("milligrams", UnitDimension.MASS),
            ("millilitre", UnitDimension.VOLUME),
            ("milliliter", UnitDimension.VOLUME),
            ("litre", UnitDimension.VOLUME),
            ("liter", UnitDimension.VOLUME),
            ("centilitre", UnitDimension.VOLUME),
            ("centiliter", UnitDimension.VOLUME),
            ("decilitre", UnitDimension.VOLUME),
            ("deciliter", UnitDimension.VOLUME),
            ("pieces", UnitDimension.COUNT),
        ],
    )
    def test_alias_unit(self, alias: str, expected: UnitDimension) -> None:
        assert UnitClassifier.classify(alias) is expected


class TestClassifyErrors:
    """Error paths for UnitClassifier.classify."""

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(UnknownUnitError, match="tablespoon"):
            UnitClassifier.classify("tablespoon")

    def test_unknown_gibberish_raises(self) -> None:
        with pytest.raises(UnknownUnitError, match="xyzzy"):
            UnitClassifier.classify("xyzzy")

    def test_none_raises(self) -> None:
        with pytest.raises(UnknownUnitError, match="None"):
            UnitClassifier.classify(None)  # type: ignore[arg-type]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(UnknownUnitError, match="non-empty"):
            UnitClassifier.classify("")

    def test_whitespace_string_raises(self) -> None:
        with pytest.raises(UnknownUnitError, match="non-empty"):
            UnitClassifier.classify("   ")

    def test_int_raises(self) -> None:
        with pytest.raises(UnknownUnitError, match="0"):
            UnitClassifier.classify(0)  # type: ignore[arg-type]

    def test_case_sensitive(self) -> None:
        # The classifier is case-sensitive; "KG" is not "kg".
        with pytest.raises(UnknownUnitError, match="KG"):
            UnitClassifier.classify("KG")


# ======================================================================
# UnitClassifier.are_compatible
# ======================================================================


class TestAreCompatible:
    """UnitClassifier.are_compatible checks."""

    @pytest.mark.parametrize(
        "a,b",
        [("g", "kg"), ("mg", "g"), ("kg", "kg"), ("ml", "l"), ("dl", "cl")],
    )
    def test_same_dimension_true(self, a: str, b: str) -> None:
        assert UnitClassifier.are_compatible(a, b) is True

    @pytest.mark.parametrize(
        "a,b",
        [("g", "ml"), ("kg", "piece"), ("l", "pcs"), ("pc", "g")],
    )
    def test_different_dimension_false(self, a: str, b: str) -> None:
        assert UnitClassifier.are_compatible(a, b) is False

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(UnknownUnitError):
            UnitClassifier.are_compatible("g", "cups")


# ======================================================================
# UnitConverter.convert — intra-dimension
# ======================================================================


class TestConvertIntraMass:
    """Intra-dimension mass conversions."""

    def test_g_to_kg(self, converter: UnitConverter) -> None:
        result = converter.convert(Decimal(1000), "g", "kg")
        assert result == Decimal(1)

    def test_kg_to_g(self, converter: UnitConverter) -> None:
        result = converter.convert(Decimal("2.5"), "kg", "g")
        assert result == Decimal(2500)

    def test_mg_to_g(self, converter: UnitConverter) -> None:
        result = converter.convert(Decimal(500), "mg", "g")
        assert result == Decimal("0.5")

    def test_g_to_mg(self, converter: UnitConverter) -> None:
        result = converter.convert(Decimal(1), "g", "mg")
        assert result == Decimal(1000)

    def test_kg_to_mg(self, converter: UnitConverter) -> None:
        result = converter.convert(Decimal("0.001"), "kg", "mg")
        assert result == Decimal(1000)

    def test_mg_to_kg(self, converter: UnitConverter) -> None:
        result = converter.convert(Decimal(1000000), "mg", "kg")
        assert result == Decimal(1)

    def test_identity(self, converter: UnitConverter) -> None:
        result = converter.convert(Decimal(42), "g", "g")
        assert result == Decimal(42)


class TestConvertIntraVolume:
    """Intra-dimension volume conversions."""

    def test_ml_to_l(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(1000), "ml", "l") == Decimal(1)

    def test_l_to_ml(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal("0.75"), "l", "ml") == Decimal(750)

    def test_cl_to_ml(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(5), "cl", "ml") == Decimal(50)

    def test_dl_to_l(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(5), "dl", "l") == Decimal("0.5")

    def test_l_to_cl(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(1), "l", "cl") == Decimal(100)


class TestConvertIntraCount:
    """Intra-dimension count conversions."""

    def test_piece_to_pc(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(3), "piece", "pc") == Decimal(3)

    def test_pc_to_pcs(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(5), "pc", "pcs") == Decimal(5)


class TestConvertIntraAliases:
    """Conversions using alias unit names."""

    def test_litre_to_ml(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(1), "litre", "ml") == Decimal(1000)

    def test_kilogram_to_g(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(1), "kilogram", "g") == Decimal(1000)

    def test_pieces_to_piece(self, converter: UnitConverter) -> None:
        assert converter.convert(Decimal(2), "pieces", "piece") == Decimal(2)


# ======================================================================
# UnitConverter.convert — cross-dimension
# ======================================================================


class TestConvertCrossDimension:
    """Cross-dimension conversions with ProductConversion."""

    def test_piece_to_g_with_conversion(
        self, converter: UnitConverter, onion_conversion: ProductConversion
    ) -> None:
        result = converter.convert(Decimal(3), "piece", "g", onion_conversion)
        assert result == Decimal(450)  # 3 × 150

    def test_fractional_piece(
        self, converter: UnitConverter, onion_conversion: ProductConversion
    ) -> None:
        result = converter.convert(Decimal("0.5"), "piece", "g", onion_conversion)
        assert result == Decimal(75)

    def test_without_conversion_raises(self, converter: UnitConverter) -> None:
        with pytest.raises(CrossDimensionError, match="ProductConversion"):
            converter.convert(Decimal(2), "piece", "g")

    def test_conversion_unit_mismatch_from(
        self, converter: UnitConverter, onion_conversion: ProductConversion
    ) -> None:
        with pytest.raises(UnitConversionError, match="from_unit"):
            converter.convert(Decimal(2), "pc", "g", onion_conversion)

    def test_conversion_unit_mismatch_to(
        self, converter: UnitConverter, onion_conversion: ProductConversion
    ) -> None:
        with pytest.raises(UnitConversionError, match="to_unit"):
            converter.convert(Decimal(2), "piece", "kg", onion_conversion)


# ======================================================================
# UnitConverter.convert — error paths
# ======================================================================


class TestConvertErrors:
    """Error handling in UnitConverter.convert."""

    def test_zero_quantity(self, converter: UnitConverter) -> None:
        with pytest.raises(InvalidQuantityError, match="> 0"):
            converter.convert(Decimal(0), "g", "kg")

    def test_negative_quantity(self, converter: UnitConverter) -> None:
        with pytest.raises(InvalidQuantityError, match="-1"):
            converter.convert(Decimal(-1), "g", "kg")

    def test_non_decimal_quantity(self, converter: UnitConverter) -> None:
        # Passing a float (not Decimal) should raise InvalidQuantityError.
        with pytest.raises(InvalidQuantityError, match="Decimal"):
            converter.convert(1.5, "g", "kg")  # type: ignore[arg-type]

    def test_unknown_from_unit(self, converter: UnitConverter) -> None:
        with pytest.raises(UnknownUnitError, match="cups"):
            converter.convert(Decimal(1), "cups", "ml")

    def test_unknown_to_unit(self, converter: UnitConverter) -> None:
        with pytest.raises(UnknownUnitError, match="oz"):
            converter.convert(Decimal(1), "g", "oz")

    def test_product_conversion_for_intra_dimension(
        self, converter: UnitConverter, onion_conversion: ProductConversion
    ) -> None:
        with pytest.raises(UnitConversionError, match="intra-dimension"):
            converter.convert(Decimal(1), "g", "kg", onion_conversion)


# ======================================================================
# ProductConversion model
# ======================================================================


class TestProductConversion:
    """ProductConversion model validation."""

    def test_create_valid(self) -> None:
        pc = ProductConversion(
            canonical_name="onion",
            from_unit="piece",
            to_unit="g",
            conversion_factor=Decimal(150),
        )
        assert pc.canonical_name == "onion"
        assert pc.from_unit == "piece"
        assert pc.to_unit == "g"
        assert pc.conversion_factor == Decimal(150)
        assert pc.source == "catalogue"

    def test_default_source(self) -> None:
        pc = ProductConversion(
            canonical_name="onion",
            from_unit="piece",
            to_unit="g",
            conversion_factor=Decimal(150),
        )
        assert pc.source == "catalogue"

    def test_custom_source(self) -> None:
        pc = ProductConversion(
            canonical_name="onion",
            from_unit="piece",
            to_unit="g",
            conversion_factor=Decimal(150),
            source="user_confirmed",
        )
        assert pc.source == "user_confirmed"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValueError):
            ProductConversion(  # type: ignore[call-arg]
                canonical_name="onion",
                from_unit="piece",
                to_unit="g",
                conversion_factor=Decimal(150),
                extra_field="should not be here",
            )

    def test_frozen(self) -> None:
        pc = ProductConversion(
            canonical_name="onion",
            from_unit="piece",
            to_unit="g",
            conversion_factor=Decimal(150),
        )
        # Pydantic frozen models raise ValidationError on mutation.
        with pytest.raises((AttributeError, ValueError)):
            pc.source = "user_confirmed"  # type: ignore[misc]

    def test_zero_factor_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProductConversion(
                canonical_name="onion",
                from_unit="piece",
                to_unit="g",
                conversion_factor=Decimal(0),
            )

    def test_negative_factor_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProductConversion(
                canonical_name="onion",
                from_unit="piece",
                to_unit="g",
                conversion_factor=Decimal(-1),
            )


# ======================================================================
# scale_ingredient
# ======================================================================


class TestScaleIngredient:
    """5.4 Serving scaling — scale_ingredient."""

    def test_scale_up(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        """2 servings → 4 servings: quantity doubles."""
        result = scale_ingredient(sample_demand, Decimal(2), Decimal(4))
        assert result.quantity == Decimal(1000)
        assert result.canonical_name == sample_demand.canonical_name

    def test_scale_down(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        """4 servings → 2 servings: quantity halves."""
        result = scale_ingredient(sample_demand, Decimal(4), Decimal(2))
        assert result.quantity == Decimal(250)

    def test_identity(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        """Same servings: quantity unchanged."""
        result = scale_ingredient(sample_demand, Decimal(2), Decimal(2))
        assert result.quantity == sample_demand.quantity

    def test_fractional_servings(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        """Fractional scaling: 3 → 7 servings."""
        result = scale_ingredient(sample_demand, Decimal(3), Decimal(7))
        # 500 * (7/3) — use quantized comparison to avoid Decimal precision drift.
        expected = Decimal(500) * Decimal(7) / Decimal(3)
        assert result.quantity.quantize(Decimal("0.0001")) == expected.quantize(
            Decimal("0.0001")
        )

    def test_does_not_mutate_input(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        """Original IngredientDemand is immutable — quantity stays."""
        original = sample_demand.quantity
        scale_ingredient(sample_demand, Decimal(2), Decimal(4))
        assert sample_demand.quantity == original

    def test_different_fields_preserved(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        """Non-quantity fields are carried through unchanged."""
        result = scale_ingredient(sample_demand, Decimal(2), Decimal(4))
        assert result.canonical_name == "chicken breast"
        assert result.unit == "g"
        assert result.confidence == Decimal("0.95")

    def test_original_servings_zero_raises(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        with pytest.raises(InvalidQuantityError, match="original_servings"):
            scale_ingredient(sample_demand, Decimal(0), Decimal(4))

    def test_original_servings_negative_raises(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        with pytest.raises(InvalidQuantityError, match="original_servings"):
            scale_ingredient(sample_demand, Decimal(-1), Decimal(4))

    def test_target_servings_zero_raises(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        with pytest.raises(InvalidQuantityError, match="target_servings"):
            scale_ingredient(sample_demand, Decimal(4), Decimal(0))

    def test_target_servings_negative_raises(
        self, converter: UnitConverter, sample_demand: IngredientDemand
    ) -> None:
        with pytest.raises(InvalidQuantityError, match="target_servings"):
            scale_ingredient(sample_demand, Decimal(4), Decimal(-1))


# ======================================================================
# Exception hierarchy
# ======================================================================


class TestExceptionHierarchy:
    """Verify exception inheritance for consistent error handling."""

    def test_unknown_unit_is_unit_conversion_error(self) -> None:
        assert issubclass(UnknownUnitError, UnitConversionError)

    def test_cross_dimension_is_unit_conversion_error(self) -> None:
        assert issubclass(CrossDimensionError, UnitConversionError)

    def test_invalid_quantity_is_unit_conversion_error(self) -> None:
        assert issubclass(InvalidQuantityError, UnitConversionError)

    def test_all_are_value_errors(self) -> None:
        assert issubclass(UnitConversionError, ValueError)
