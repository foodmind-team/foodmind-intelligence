"""Domain allow-list configuration (handbook 10.4).

Maintains three separate source classes:
1. Government/food-safety sources for hard safety facts.
2. Established recipe/culinary sources for technique estimates.
3. Controlled seed catalogue.

Technique sources cannot override the hard-safety table.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar


class SourceClass(StrEnum):
    """Source trust tier — higher ordinal = stronger precedence."""

    SAFETY = "safety"        # Government / food-safety authorities
    TECHNIQUE = "technique"  # Established recipe / culinary sources
    SEED = "seed"            # Controlled seed catalogue (internal)


# ---------------------------------------------------------------------------
# Default allow-list — replaceable via Settings.allowed_research_domains
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainAllowList:
    """Immutable allow-list initialised once from Settings.

    Maps domain -> SourceClass. A domain NOT in this mapping is rejected.
    """

    # Technique sources can never override safety sources (handbook 10.4)
    domains: dict[str, SourceClass] = field(default_factory=dict)

    # Hard-safety domains (handbook 10.4 class 1)
    SAFETY_DEFAULTS: ClassVar[tuple[str, ...]] = (
        "fsai.ie",          # Food Safety Authority of Ireland
        "fda.gov",          # US Food and Drug Administration
        "food.gov.uk",      # UK Food Standards Agency
        "who.int",          # World Health Organization
        "cdc.gov",          # US Centers for Disease Control
    )

    # Established recipe/culinary domains (handbook 10.4 class 2)
    TECHNIQUE_DEFAULTS: ClassVar[tuple[str, ...]] = (
        "seriouseats.com",
        "bbcgoodfood.com",
        "allrecipes.com",
        "foodnetwork.com",
        "bonappetit.com",
    )

    @classmethod
    def from_settings(
        cls,
        custom_domains: list[str],
    ) -> "DomainAllowList":
        """Build allow-list from Settings + class-level defaults.

        Custom domains from Settings are treated as TECHNIQUE tier.
        SAFETY defaults are always included and cannot be removed.
        """
        domains: dict[str, SourceClass] = {}

        # Safety domains — always present, highest precedence
        for domain in cls.SAFETY_DEFAULTS:
            domains[domain] = SourceClass.SAFETY

        # Technique defaults
        for domain in cls.TECHNIQUE_DEFAULTS:
            domains[domain] = SourceClass.TECHNIQUE

        # User-specified custom domains (technique tier)
        for domain in custom_domains:
            # Don't override safety domains with user input
            if domain not in domains:
                domains[domain] = SourceClass.TECHNIQUE

        return cls(domains=domains)

    def classify(self, url: str) -> SourceClass | None:
        """Return the source class for a URL, or None if not allow-listed.

        Performs substring matching: a URL matches if any allowed domain
        appears as a substring of the URL host.
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
        except (ValueError, TypeError):
            return None

        for domain, source_class in self.domains.items():
            if domain in host:
                return source_class
        return None

    def is_allowed(self, url: str) -> bool:
        """Return True if the URL matches any allow-listed domain."""
        return self.classify(url) is not None
