# =============================================================================
# 区域食品安全策略包（safety/policy）
# -----------------------------------------------------------------------------
# 本文件定义区域食品安全策略包的 Schema（PolicySource / SafetyThresholds /
# SafetyPolicy）以及策略注册与解析逻辑。它将原先硬编码在安全规则中的
# 温度 / 保温 / 静置常量迁移到带版本、有来源依据、可按区域选择的策略包中。
# =============================================================================

"""Regional food-safety policy packs (P3-04).

区域食品安全策略包（P3-04）。

Migrates hard-coded temperature / holding / rest constants out of the safety
rules into versioned, source-backed, region-selectable policy packs.

将原先硬编码在安全规则中的温度 / 保温 / 静置常量迁移到带版本、有来源依据、
可按区域选择的策略包中。

Design contract (development plan P3-04, data-flow rules D6/D7):
  - Region selection is EXPLICIT: the request region overrides the deployment
    default (Settings.safety_policy_region). An unknown region is rejected —
    never silently falls back to another pack.
  - Thresholds come ONLY from approved, reviewed packs in this directory.
    LLM or web-search results can never modify them (D7).
  - Policies are versioned and immutable. New guidance ships as a NEW version;
    old versions stay registered so historical checkpoints remain auditable.
  - A policy that is not yet effective, has no sources, or belongs to an
    unknown region can never be applied to a plan (cannot enter READY).

设计契约（开发计划 P3-04，数据流规则 D6/D7）：
  - 区域选择是显式的：请求区域覆盖部署默认值（Settings.safety_policy_region）。
    未知区域会被拒绝 —— 绝不静默回退到其他策略包。
  - 阈值只来源于本目录中经批准、审查过的策略包。
    LLM 或网页搜索结果永远无法修改它们（D7）。
  - 策略带版本且不可变。新指引以新版本发布；
    旧版本保持注册，使历史检查点仍可审计。
  - 尚未生效、没有来源或属于未知区域的策略永远无法应用于计划
    （无法进入 READY）。

Resolution semantics (resolve_policy):
  1. region must be registered — otherwise UnknownRegionError.
  2. version defaults to the latest registered version of that region.
     An explicit version must exist — otherwise UnknownPolicyVersionError.
  3. effective_at must be <= today — otherwise PolicyNotYetEffectiveError.
  4. sources must be non-empty — otherwise PolicyMissingSourcesError.

解析语义（resolve_policy）：
  1. region 必须已注册 —— 否则抛出 UnknownRegionError。
  2. version 默认取该区域最新注册版本。
     显式指定的版本必须存在 —— 否则抛出 UnknownPolicyVersionError。
  3. effective_at 必须不晚于今天 —— 否则抛出 PolicyNotYetEffectiveError。
  4. sources 必须非空 —— 否则抛出 PolicyMissingSourcesError。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from cooking_plan_agent.domain.models import PolicySourceRef, SafetyPolicyRecord

# =============================================================================
# Schema
# 数据结构定义
# =============================================================================


@dataclass(frozen=True)
class PolicySource:
    """An official source backing one or more thresholds (D7 traceability). 支撑一个或多个阈值的官方来源（D7 可追溯性）。"""

    source_id: str
    title: str
    url: str


@dataclass(frozen=True)
class SafetyThresholds:
    """Region-specific safety thresholds consumed by the safety rules.

    安全规则所消费的区域专属安全阈值。

    Every value is locked by unit-test fixtures that assert the exact number
    and its source provenance (plan P3-04 verification).

    每个值都由单元测试夹具锁定，断言确切的数值及其来源出处（计划 P3-04 验证）。

    ``safe_minimum_temperatures_c`` keys the per-protein safe minimum internal
    cooking temperatures in °C (same protein categories as the keyword matcher
    in safety/rules.py). Categories absent from the map are "not documented by
    this authority" and are NOT flagged by ProteinSafetyTemperatureRule.

    ``safe_minimum_temperatures_c`` 以每种蛋白质的安全最低内部烹饪温度（°C）
    为键（与 safety/rules.py 中关键词匹配器的蛋白质类别一致）。
    表中缺失的类别表示“该机构未记录”，ProteinSafetyTemperatureRule 不会标记。
    """

    safe_minimum_temperatures_c: dict[str, Decimal]
    # Max time perishable food may sit in the danger zone (room temperature).
    # 易腐食品可处于危险温度区（室温）的最长时间。
    max_room_temp_holding_minutes: int
    # Hot holding: keep cooked food at or above this temperature.
    # 热保温：熟食应保持在等于或高于此温度。
    hot_holding_minimum_c: Decimal
    # Cold holding: keep chilled food at or below this temperature.
    # 冷保温：冷藏食品应保持在等于或低于此温度。
    cold_holding_maximum_c: Decimal
    # Reheating: reach at least this internal temperature …
    # 复热：至少达到此内部温度……
    reheat_minimum_c: Decimal
    # … and hold it for at least this long (seconds). 0 = no hold requirement.
    # ……并至少保持这么长时间（秒）。0 = 无保持要求。
    reheat_hold_seconds: int
    # Post-cooking rest time for protein categories that need it (minutes).
    # 需要静置的蛋白质类别在烹饪后的静置时间（分钟）。
    rest_time_minutes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyPolicy:
    """An immutable, versioned, source-backed safety policy pack.

    一个不可变、带版本、有来源依据的食品安全策略包。

    ``region`` is a stable ISO-3166 alpha-2 code ("US", "SG"). ``version``
    follows MAJOR.MINOR and is bumped (never mutated) when thresholds change.

    ``region`` 是稳定的 ISO-3166 alpha-2 代码（"US"、"SG"）。
    ``version`` 遵循 MAJOR.MINOR，并在阈值变化时递增（从不原地修改）。
    """

    region: str
    version: str
    effective_at: date
    sources: tuple[PolicySource, ...]
    thresholds: SafetyThresholds

    def to_record(self) -> SafetyPolicyRecord:
        """Project this policy into the serialisable domain record.

        将本策略投影为可序列化的领域记录。

        The record is what travels in workflow state and terminal responses —
        the full SafetyPolicy (with rule-config thresholds) stays in-process.

        该记录在工作流状态与终态响应中传递 ——
        完整的 SafetyPolicy（含规则配置阈值）则保持在进程内部。
        """
        return SafetyPolicyRecord(
            region=self.region,
            version=self.version,
            effective_at=self.effective_at,
            sources=tuple(PolicySourceRef(source_id=s.source_id, title=s.title, url=s.url) for s in self.sources),
        )


# =============================================================================
# Registry & resolution
# 注册表与解析
# =============================================================================


class PolicyResolutionError(ValueError):
    """Base class for policy resolution failures (P3-04 D6: no silent fallback). 策略解析失败的基类（P3-04 D6：禁止静默回退）。"""


class UnknownRegionError(PolicyResolutionError):
    """The requested region has no registered policy pack. 请求的区域没有已注册的策略包。"""


class UnknownPolicyVersionError(PolicyResolutionError):
    """The requested version is not registered for the region. 请求的版本未在该区域注册。"""


class PolicyNotYetEffectiveError(PolicyResolutionError):
    """The policy's effective_at date is in the future. 策略的生效日期（effective_at）在未来。"""


class PolicyMissingSourcesError(PolicyResolutionError):
    """The policy declares no official sources (unverifiable thresholds). 策略未声明官方来源（阈值无法验证）。"""


# Registered packs keyed by (region, version). Regions MUST be upper-case.
# 已注册策略包，以 (region, version) 为键。区域必须大写。
_POLICY_REGISTRY: dict[tuple[str, str], SafetyPolicy] = {}


def register_policy(policy: SafetyPolicy) -> None:
    """Register a policy pack (called once at import time by policies/__init__).

    注册一个策略包（由 policies/__init__ 在导入时调用一次）。

    Registration replaces nothing silently: a duplicate (region, version) pair
    raises ValueError so packs cannot be overwritten by accident.

    注册不会静默替换任何内容：重复的 (region, version) 组合会抛出 ValueError，
    以免策略包被意外覆盖。
    """
    key = (policy.region.upper(), policy.version)
    if key in _POLICY_REGISTRY:
        raise ValueError(f"Policy already registered: region={key[0]} version={key[1]}")
    _POLICY_REGISTRY[key] = policy


def supported_regions() -> tuple[str, ...]:
    """Return the sorted regions that have at least one registered pack. 返回至少有一个已注册策略包的、经过排序的区域列表。"""
    return tuple(sorted({region for region, _ in _POLICY_REGISTRY}))


def latest_version(region: str) -> str | None:
    """Return the highest registered version for a region (None if unknown). 返回某区域已注册的最高版本（若未知则返回 None）。"""
    versions = [ver for reg, ver in _POLICY_REGISTRY if reg == region.upper()]
    if not versions:
        return None
    return max(versions)


def resolve_policy(region: str, version: str | None = None) -> SafetyPolicy:
    """Resolve the active policy for a region, honouring the P3-04 reject rules.

    为某区域解析生效策略，遵守 P3-04 的拒绝规则。

    Args:
        region: Explicit region (ISO alpha-2, case-insensitive). Never None —
                callers must supply the request region or the deployment
                default; an unknown region is a hard error (D6).
        region：显式区域（ISO alpha-2，大小写不敏感）。绝不传 None ——
                调用方必须提供请求区域或部署默认值；未知区域为硬错误（D6）。
        version: Optional explicit version. Defaults to the latest registered
                 version of the region.
        version：可选的显式版本。默认取该区域最新注册版本。

    Returns:
        The resolved SafetyPolicy.
        解析得到的 SafetyPolicy。

    Raises:
        UnknownRegionError: region has no registered pack.
        UnknownRegionError：区域没有已注册的策略包。
        UnknownPolicyVersionError: explicit version not registered.
        UnknownPolicyVersionError：显式版本未注册。
        PolicyNotYetEffectiveError: policy not yet effective.
        PolicyNotYetEffectiveError：策略尚未生效。
        PolicyMissingSourcesError: policy has no official sources.
        PolicyMissingSourcesError：策略没有官方来源。
    """
    region_key = region.upper()
    if region_key not in {r for r, _ in _POLICY_REGISTRY}:
        raise UnknownRegionError(
            f"Unknown safety-policy region: {region!r}. Supported regions: {', '.join(supported_regions()) or '(none)'}"
        )

    if version is None:
        version = latest_version(region_key) or ""
    key = (region_key, version)
    policy = _POLICY_REGISTRY.get(key)
    if policy is None:
        raise UnknownPolicyVersionError(
            f"No safety policy for region {region_key!r} version {version!r}. "
            f"Available versions: {[v for r, v in _POLICY_REGISTRY if r == region_key]}"
        )

    if policy.effective_at > date.today():
        raise PolicyNotYetEffectiveError(
            f"Safety policy {region_key}@{policy.version} is not effective until {policy.effective_at.isoformat()}"
        )

    if not policy.sources:
        raise PolicyMissingSourcesError(
            f"Safety policy {region_key}@{policy.version} declares no official sources — cannot be applied"
        )

    return policy
