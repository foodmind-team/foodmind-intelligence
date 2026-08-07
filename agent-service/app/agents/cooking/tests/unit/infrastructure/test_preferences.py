"""P5-4: 长期偏好存储（PreferenceStore）。"""

from pathlib import Path

from cooking_plan_agent.infrastructure.preferences import PreferenceStore
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.input_parsing import merge_preferences_into_request


def _store(tmp_path: Path) -> PreferenceStore:
    return PreferenceStore(tmp_path / "prefs.sqlite")


def test_unknown_user_returns_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get("nobody") == {}


def test_write_read_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = {"dietary_restrictions": ["vegetarian"], "allergens": ["peanut"]}
    store.put("u1", payload)
    assert store.get("u1") == payload


def test_put_overwrites_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.put("u1", {"dietary_restrictions": ["vegan"]})
    store.put("u1", {"dietary_restrictions": ["vegetarian"], "allergens": ["shellfish"]})
    assert store.get("u1") == {"dietary_restrictions": ["vegetarian"], "allergens": ["shellfish"]}


def test_merge_preferences_explicit_wins(tmp_path: Path) -> None:
    """显式请求值优先于记忆值（用户当场确认的信息覆盖长期偏好）。"""
    store = _store(tmp_path)
    store.put("u1", {"dietary_restrictions": ["vegan"], "allergens": ["peanut"]})

    request = {
        "request_id": "r1",
        "user_id": "u1",
        "recipes": ({"recipe_id": "r1", "text": "x", "target_servings": 2},),
        "dietary_restrictions": ("vegetarian",),  # 显式值，覆盖记忆里的 vegan
    }
    merged = merge_preferences_into_request(request, store.get("u1"))
    assert merged["dietary_restrictions"] == ("vegetarian",)
    # 记忆里未被显式覆盖的字段被注入。
    assert merged["user_allergens"] == ("peanut",)


def test_merge_preferences_without_user_id_is_noop(tmp_path: Path) -> None:
    """无 user_id 时不注入任何记忆（零回归）。"""
    store = _store(tmp_path)
    store.put("u1", {"dietary_restrictions": ["vegan"]})

    request = {
        "request_id": "r1",
        "recipes": ({"recipe_id": "r1", "text": "x", "target_servings": 2},),
        "dietary_restrictions": (),
    }
    merged = merge_preferences_into_request(request, store.get("nobody"))
    assert merged["dietary_restrictions"] == ()
    assert "user_allergens" not in merged or merged["user_allergens"] == ()


def test_context_accepts_preference_store(tmp_path: Path) -> None:
    """WorkflowContext 可注入可选 PreferenceStore（DI）。"""
    store = _store(tmp_path)
    ctx = WorkflowContext(recipe_extractor=None, preference_store=store)
    assert ctx.preference_store is not None
