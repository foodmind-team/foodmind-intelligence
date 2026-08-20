# =============================================================================
# 菜谱缺口补全辅助模块（research 包）
# -----------------------------------------------------------------------------
# 提供“缺口 → 证据 → 应用”链路中的辅助函数。联网搜索已移除，
# 缺口补全完全依赖 LLM 烹饪知识（llm/researcher.py）。
# =============================================================================

"""Recipe-gap completion helpers (gap → evidence → apply).

菜谱缺口补全辅助模块（缺口 → 证据 → 应用）。

Web search was removed; gap completion now relies solely on LLM culinary
knowledge (``llm/researcher.py``). The remaining modules here support that
path:

联网搜索已移除；缺口补全现在完全依赖 LLM 烹饪知识（``llm/researcher.py``）。
本目录下其余模块支撑这条路径：

- ``query_builder``: build a bounded, privacy-safe gap query.
- ``reconciler``: reconcile multiple evidence items into consensus.
- ``evidence_apply``: write reconciled evidence back into candidates.

- ``query_builder``：构建有界且保护隐私的缺口查询。
- ``reconciler``：将多条证据调和为共识。
- ``evidence_apply``：将调和后的证据写回候选菜谱。
"""
