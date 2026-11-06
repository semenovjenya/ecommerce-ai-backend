from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.graph import build_graph
from commerce_agent.serialization import checkpoint_serializer


def test_strict_checkpoint_serializer_round_trip() -> None:
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": "strict-serialization"}}
    context = CommerceRuntimeContext(
        tenant_id="tenant-1",
        user_id="user-1",
        roles=("manager",),
        request_id="request-1",
    )

    result = graph.invoke(
        {
            "schema_version": 1,
            "messages": [HumanMessage(content="strict restore", id="message-1")],
        },
        config=config,
        context=context,
    )
    restored = graph.get_state(config)
    history = list(graph.get_state_history(config))

    assert result["schema_version"] == 1
    assert restored.values["schema_version"] == 1
    assert restored.values["sales_analysis"].domain.value == "sales"
    assert len(history) > 1
    assert all(snapshot.values["schema_version"] == 1 for snapshot in history)
