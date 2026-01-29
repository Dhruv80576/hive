"""
Tests for cycle detection in graph validation.
"""

import pytest

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.node import NodeSpec


def test_no_cycle_linear_graph():
    """Test that a simple linear graph has no cycles."""
    graph = GraphSpec(
        id="linear-graph",
        goal_id="test-goal",
        entry_node="A",
        terminal_nodes=["C"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Test node A", node_type="llm_generate"),
            NodeSpec(id="B", name="Node B", description="Test node B", node_type="llm_generate"),
            NodeSpec(id="C", name="Node C", description="Test node C", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="B", condition=EdgeCondition.ALWAYS),
            EdgeSpec(id="e2", source="B", target="C", condition=EdgeCondition.ALWAYS),
        ],
    )
    
    errors = graph.validate()
    assert not errors, f"Expected no errors, got: {errors}"


def test_simple_cycle_detected():
    """Test that a simple A->B->A cycle is detected."""
    graph = GraphSpec(
        id="cycle-graph",
        goal_id="test-goal",
        entry_node="A",
        terminal_nodes=["B"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Test node A", node_type="llm_generate"),
            NodeSpec(id="B", name="Node B", description="Test node B", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="B", condition=EdgeCondition.ALWAYS),
            EdgeSpec(id="e2", source="B", target="A", condition=EdgeCondition.ALWAYS),
        ],
    )
    
    errors = graph.validate()
    assert len(errors) == 1
    assert "Cycle detected" in errors[0]
    assert "Node A → Node B → Node A" in errors[0]


def test_self_loop_detected():
    """Test that a node pointing to itself is detected as a cycle."""
    graph = GraphSpec(
        id="self-loop-graph",
        goal_id="test-goal",
        entry_node="A",
        terminal_nodes=["A"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Test node A", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="A", condition=EdgeCondition.ALWAYS),
        ],
    )
    
    errors = graph.validate()
    assert len(errors) == 1
    assert "Cycle detected" in errors[0]
    assert "Node A → Node A" in errors[0]


def test_complex_cycle_detected():
    """Test that a cycle in a more complex graph is detected."""
    graph = GraphSpec(
        id="complex-cycle-graph",
        goal_id="test-goal",
        entry_node="A",
        terminal_nodes=["D"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Test node A", node_type="llm_generate"),
            NodeSpec(id="B", name="Node B", description="Test node B", node_type="llm_generate"),
            NodeSpec(id="C", name="Node C", description="Test node C", node_type="llm_generate"),
            NodeSpec(id="D", name="Node D", description="Test node D", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="B", condition=EdgeCondition.ALWAYS),
            EdgeSpec(id="e2", source="B", target="C", condition=EdgeCondition.ALWAYS),
            EdgeSpec(id="e3", source="C", target="B", condition=EdgeCondition.ALWAYS),  # Cycle B->C->B
            EdgeSpec(id="e4", source="C", target="D", condition=EdgeCondition.ON_FAILURE),
        ],
    )
    
    errors = graph.validate()
    assert len(errors) >= 1
    cycle_errors = [e for e in errors if "Cycle detected" in e]
    assert len(cycle_errors) == 1
    assert "Node B → Node C → Node B" in cycle_errors[0]


def test_branching_no_cycle():
    """Test that branching without cycles is valid."""
    graph = GraphSpec(
        id="branching-graph",
        goal_id="test-goal",
        entry_node="A",
        terminal_nodes=["C", "D"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Router node", node_type="router"),
            NodeSpec(id="B", name="Node B", description="Test node B", node_type="llm_generate"),
            NodeSpec(id="C", name="Node C", description="Test node C", node_type="llm_generate"),
            NodeSpec(id="D", name="Node D", description="Test node D", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="B", condition=EdgeCondition.ON_SUCCESS),
            EdgeSpec(id="e2", source="A", target="C", condition=EdgeCondition.ON_FAILURE),
            EdgeSpec(id="e3", source="B", target="D", condition=EdgeCondition.ALWAYS),
        ],
    )
    
    errors = graph.validate()
    assert not errors, f"Expected no errors, got: {errors}"


def test_convergence_no_cycle():
    """Test that convergence (fan-in) without cycles is valid."""
    graph = GraphSpec(
        id="convergence-graph",
        goal_id="test-goal",
        entry_node="A",
        terminal_nodes=["D"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Router node", node_type="router"),
            NodeSpec(id="B", name="Node B", description="Test node B", node_type="llm_generate"),
            NodeSpec(id="C", name="Node C", description="Test node C", node_type="llm_generate"),
            NodeSpec(id="D", name="Node D", description="Test node D", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="B", condition=EdgeCondition.ON_SUCCESS),
            EdgeSpec(id="e2", source="A", target="C", condition=EdgeCondition.ON_SUCCESS),
            EdgeSpec(id="e3", source="B", target="D", condition=EdgeCondition.ALWAYS),
            EdgeSpec(id="e4", source="C", target="D", condition=EdgeCondition.ALWAYS),
        ],
    )
    
    errors = graph.validate()
    assert not errors, f"Expected no errors, got: {errors}"


def test_multiple_entry_points_no_cycle():
    """Test that multiple entry points without cycles is valid."""
    graph = GraphSpec(
        id="multi-entry-graph",
        goal_id="test-goal",
        entry_node="A",
        entry_points={"resume": "B"},
        terminal_nodes=["C"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Test node A", node_type="llm_generate"),
            NodeSpec(id="B", name="Node B", description="Test node B", node_type="llm_generate"),
            NodeSpec(id="C", name="Node C", description="Test node C", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="B", condition=EdgeCondition.ALWAYS),
            EdgeSpec(id="e2", source="B", target="C", condition=EdgeCondition.ALWAYS),
        ],
    )
    
    errors = graph.validate()
    assert not errors, f"Expected no errors, got: {errors}"


def test_conditional_cycle_detected():
    """Test that cycles with conditional edges are still detected."""
    graph = GraphSpec(
        id="conditional-cycle-graph",
        goal_id="test-goal",
        entry_node="A",
        terminal_nodes=["C"],
        nodes=[
            NodeSpec(id="A", name="Node A", description="Test node A", node_type="llm_generate", output_keys=["result"]),
            NodeSpec(id="B", name="Node B", description="Test node B", node_type="llm_generate"),
            NodeSpec(id="C", name="Node C", description="Test node C", node_type="llm_generate"),
        ],
        edges=[
            EdgeSpec(id="e1", source="A", target="B", condition=EdgeCondition.ALWAYS),
            EdgeSpec(
                id="e2",
                source="B",
                target="A",
                condition=EdgeCondition.CONDITIONAL,
                condition_expr="result < 5",
            ),
            EdgeSpec(id="e3", source="B", target="C", condition=EdgeCondition.ON_SUCCESS),
        ],
    )
    
    errors = graph.validate()
    assert len(errors) >= 1
    cycle_errors = [e for e in errors if "Cycle detected" in e]
    assert len(cycle_errors) == 1
    assert "Node A → Node B → Node A" in cycle_errors[0]
