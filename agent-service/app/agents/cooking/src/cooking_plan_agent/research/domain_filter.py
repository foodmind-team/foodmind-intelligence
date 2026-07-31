"""Domain allow-list filtering (handbook 10.4).

Enforces that search results come only from trusted domains.
Technique sources cannot override safety sources.
"""

from cooking_plan_agent.domain.models import SearchDocument
from cooking_plan_agent.research.config import DomainAllowList, SourceClass


def filter_by_domain(
    documents: tuple[SearchDocument, ...],
    allow_list: DomainAllowList,
) -> tuple[SearchDocument, ...]:
    """Return only documents whose URLs match the allow-list.

    Documents from non-allow-listed domains are silently dropped.
    """
    return tuple(d for d in documents if allow_list.is_allowed(d.url))


def classify_documents(
    documents: tuple[SearchDocument, ...],
    allow_list: DomainAllowList,
) -> tuple[tuple[SearchDocument, ...], tuple[SearchDocument, ...]]:
    """Split documents into safety and technique classes.

    Returns (safety_docs, technique_docs). Technique sources
    never override safety sources (handbook 10.4).
    """
    safety_docs: list[SearchDocument] = []
    technique_docs: list[SearchDocument] = []

    for doc in documents:
        source_class = allow_list.classify(doc.url)
        if source_class == SourceClass.SAFETY:
            safety_docs.append(doc)
        elif source_class == SourceClass.TECHNIQUE:
            technique_docs.append(doc)
        # SEED and unclassified are ignored at this level

    return tuple(safety_docs), tuple(technique_docs)
