from workflow_helpers import workflow_context

from recommendation_agent.workflow.graph import MAX_WORKFLOW_STEPS, NODE_ORDER, BoundedRecommendationWorkflow


def test_compiled_graph_has_fixed_nodes_edges_and_step_bound() -> None:
    workflow = BoundedRecommendationWorkflow(workflow_context())
    graph = workflow.compiled.get_graph()
    assert set(graph.nodes) == {"__start__", "__end__", *NODE_ORDER, "build_failure"}
    assert MAX_WORKFLOW_STEPS == 8
    assert sum(1 for node in NODE_ORDER if node == "score_once") == 1
    assert all(edge.source != edge.target for edge in graph.edges)
