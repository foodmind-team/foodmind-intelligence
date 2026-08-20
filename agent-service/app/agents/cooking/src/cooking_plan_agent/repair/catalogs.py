# =============================================================================
# 静态修复选项目录模块（repair/catalogs）
# -----------------------------------------------------------------------------
# 保存食材替代与设备替代的静态目录，与提议行为解耦。
# =============================================================================

"""Static repair-option catalogs, kept separate from proposal behavior.

静态修复选项目录，与提议行为分开存放。
"""

# 食材替代目录：食材名 → ((替代食材, 说明), ...)
_INGREDIENT_SUBSTITUTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "chicken breast": (
        ("chicken thigh", "same protein, slightly longer cook time"),
        ("tofu (firm)", "vegetarian alternative, absorbs flavors well"),
    ),
    "chicken thigh": (
        ("chicken breast", "leaner, slightly shorter cook time"),
        ("tofu (firm)", "vegetarian alternative"),
    ),
    "beef": (
        ("pork", "similar texture for most dishes"),
        ("lamb", "richer flavor, works in stews and roasts"),
        ("tofu (extra firm)", "vegetarian alternative for stir-fries"),
    ),
    "pork": (
        ("chicken", "leaner, adjust cooking time"),
        ("beef", "richer, similar cooking methods"),
    ),
    "fish": (
        ("shrimp", "similar light protein"),
        ("tofu (silken)", "vegetarian alternative for steamed dishes"),
    ),
    "salmon": (
        ("trout", "similar fatty fish"),
        ("mackerel", "stronger flavor, similar cooking methods"),
    ),
    "shrimp": (
        ("scallop", "similar delicate seafood"),
        ("tofu (firm, cubed)", "vegetarian alternative"),
    ),
    "egg": (
        ("flax egg (1 tbsp flax + 3 tbsp water)", "vegan baking substitute"),
        ("mashed banana (1/4 cup per egg)", "vegan baking substitute"),
    ),
    "milk": (
        ("oat milk", "dairy-free, similar creaminess"),
        ("soy milk", "dairy-free, high protein"),
        ("water (with extra fat)", "emergency substitute"),
    ),
    "butter": (
        ("vegetable oil (reduce by 20%)", "dairy-free, works in most recipes"),
        ("margarine", "direct substitute"),
        ("coconut oil", "dairy-free, adds subtle coconut flavor"),
    ),
    "cream": (
        ("coconut cream", "dairy-free, similar richness"),
        ("cashew cream", "dairy-free, neutral flavor"),
    ),
    "cheese": (
        ("nutritional yeast (for flavor)", "dairy-free cheese flavor substitute"),
        ("vegan cheese", "direct dairy-free substitute"),
    ),
    "wheat flour": (
        ("almond flour", "gluten-free, adjust liquid"),
        ("rice flour", "gluten-free, works for coatings"),
        ("gluten-free flour blend", "direct gluten-free substitute"),
    ),
    "tomato": (
        ("bell pepper", "different flavor, similar texture for sauces"),
        ("canned tomato", "more concentrated, adjust liquid"),
    ),
    "onion": (
        ("shallot", "milder, more delicate flavor"),
        ("leek", "milder, works in soups and stews"),
        ("onion powder (1 tbsp = 1 medium onion)", "concentrated flavor"),
    ),
    "garlic": (
        ("garlic powder (1/8 tsp = 1 clove)", "concentrated, milder"),
        ("shallot", "different flavor, similar aromatic role"),
    ),
    "soy sauce": (
        ("tamari", "gluten-free, similar flavor"),
        ("coconut aminos", "soy-free, slightly sweeter"),
        ("fish sauce (use less)", "stronger umami, different base"),
    ),
    "rice": (
        ("quinoa", "higher protein, similar cooking time"),
        ("couscous", "faster cooking, different texture"),
        ("cauliflower rice", "low-carb alternative"),
    ),
    "pasta": (
        ("zucchini noodles", "low-carb, shorter cook time"),
        ("rice noodles", "gluten-free, different texture"),
        ("gluten-free pasta", "direct gluten-free substitute"),
    ),
    "sugar": (
        ("honey (reduce liquid by 1/4)", "natural sweetener, stronger flavor"),
        ("maple syrup (reduce liquid by 1/4)", "natural sweetener"),
    ),
    "olive oil": (
        ("vegetable oil", "neutral, higher smoke point"),
        ("avocado oil", "similar quality, higher smoke point"),
        ("coconut oil", "adds coconut flavor, solid at room temp"),
    ),
    "cilantro": (
        ("parsley + lime zest", "similar fresh note without soapy flavor"),
        ("Thai basil", "different but complementary flavor"),
    ),
    "chilli": (
        ("bell pepper + cayenne", "milder heat"),
        ("jalapeño", "different heat profile"),
    ),
}

# 设备替代目录：设备名 → ((替代设备, 说明), ...)
_EQUIPMENT_ALTERNATIVES: dict[str, tuple[tuple[str, str], ...]] = {
    "oven": (
        ("air fryer", "faster, similar results for most baked dishes"),
        ("toaster oven", "works for small batches"),
        ("stove + covered pot", "simulates oven for stews and braises"),
    ),
    "stove": (
        ("electric skillet", "portable, similar temperature control"),
        ("induction cooktop", "portable, rapid heating"),
        ("camping stove", "portable, suitable for basic cooking"),
    ),
    "wok": (
        ("large frying pan", "works for most stir-fry dishes"),
        ("cast iron skillet", "excellent heat retention for high-heat cooking"),
    ),
    "steamer": (
        ("pot + colander", "classic improvised steamer"),
        ("microwave + covered bowl", "faster for vegetables"),
    ),
    "blender": (
        ("food processor", "works for most blending tasks"),
        ("immersion blender", "works directly in pot/bowl"),
        ("mortar and pestle", "manual alternative for pastes and small batches"),
    ),
    "mixing_bowl": (
        ("large pot", "substitute for mixing larger quantities"),
        ("any large container", "temporary substitute"),
    ),
    "cutting_board": (
        ("clean countertop + silicone mat", "temporary substitute"),
        ("large flat plate", "works for small prep tasks"),
    ),
    "spatula": (
        ("wooden spoon", "works for most stirring tasks"),
        ("chopsticks", "works for stir-frying small portions"),
    ),
    "sink": (
        ("large bowl/basin", "manual washing substitute"),
        ("bathtub", "emergency substitute for large items"),
    ),
    "knife": (
        ("kitchen shears", "works for cutting herbs and small items"),
        ("mandoline slicer", "works for uniform slicing"),
    ),
    "rice_cooker": (
        ("pot with lid", "standard stovetop rice method"),
        ("instant pot", "pressure-cooker rice method"),
    ),
    "slow_cooker": (
        ("dutch oven + low oven", "simulates slow cooking"),
        ("pressure cooker", "faster, similar tenderizing"),
    ),
    "grill": (
        ("broiler (oven)", "similar high-heat from above"),
        ("grill pan (stove)", "indoor alternative with grill marks"),
        ("cast iron skillet", "excellent searing similar to grill"),
    ),
    "microwave": (
        ("stove + small pot", "reheating and steaming alternative"),
        ("oven at low temp", "gentle reheating"),
    ),
    "thermometer": (
        ("visual cues + timing", "less precise but workable for experienced cooks"),
        ("touch test (for meats)", "traditional method"),
    ),
}
