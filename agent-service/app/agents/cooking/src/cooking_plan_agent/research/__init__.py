"""Recipe-gap completion helpers (gap → evidence → apply).

Web search was removed; gap completion now relies solely on LLM culinary
knowledge (``llm/researcher.py``). The remaining modules here support that
path:

- ``query_builder``: build a bounded, privacy-safe gap query.
- ``reconciler``: reconcile multiple evidence items into consensus.
- ``evidence_apply``: write reconciled evidence back into candidates.
"""
